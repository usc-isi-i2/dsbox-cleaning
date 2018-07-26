import logging
from typing import NamedTuple, Dict, List, Set, Union

import d3m
import d3m.metadata.base as mbase
import numpy as np
import pandas as pd
from common_primitives import utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams as metadata_hyperparams
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import Enumeration, UniformInt, UniformBool
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from . import config

_logger = logging.getLogger(__name__)

Input = d3m.container.DataFrame
Output = d3m.container.DataFrame


class EncParams(params.Params):
    mapping: Dict
    cat_columns: List[str]
    empty_columns: List[int]


class EncHyperparameter(hyperparams.Hyperparams):
    n_limit = UniformInt(lower=5, upper=100, default=12,
                         description='Limits the maximum number of columns generated from a single categorical column',
                         semantic_types=['http://schema.org/Integer',
                                         'https://metadata.datadrivendiscovery.org/types/TuningParameter'])


class Encoder(UnsupervisedLearnerPrimitiveBase[Input, Output, EncParams, EncHyperparameter]):
    """
    An one-hot encoder, which
    1. n_limit: max number of distinct values to one-hot encode,
         remaining values with fewer occurence are put in [colname]_other_ column.

    2. feed in data by set_training_data, then apply fit() function to tune the encoder.

    3. produce(): input data would be encoded and return.
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "18f0bb42-6350-3753-8f2d-d1c3da70f279",
        "version": config.VERSION,
        "name": "ISI DSBox Data Encoder",
        "description": "Encode data, such as one-hot encoding for categorical data",
        "python_path": "d3m.primitives.dsbox.Encoder",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["ENCODE_ONE_HOT"],  # !!!! Need to submit algorithm type "Imputation"
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["preprocessing", "encoding"],
        "installation": [config.INSTALLATION],
    })

    def __repr__(self):
        return "%s(%r)" % ('Encoder', self.__dict__)

    def __init__(self, *, hyperparams: EncHyperparameter) -> None:

        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._mapping: Dict = {}
        self._input_data: Input = None
        self._input_data_copy = None
        self._fitted = False
        self._cat_columns = []
        self._col_index = None
        self._empty_columns = []

    def set_training_data(self, *, inputs: Input) -> None:
        self._input_data = inputs
        self._fitted = False

    def _trim_features(self, feature, n_limit):

        topn = feature.dropna().unique()
        if n_limit:
            if feature.dropna().nunique() > n_limit:
                topn = list(feature.value_counts().head(n_limit).index)
                topn.append('other_')
        topn = [x for x in topn if x]
        return feature.name, topn

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:

        if self._fitted:
            return

        if self._input_data is None:
            raise ValueError('Missing training(fitting) data.')

        # Look at attribute columns only
        # print('fit in', self._input_data.columns)
        data = self._input_data.copy()
        all_attributes = utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/Attribute"])

        # Remove columns with all empty values, structural type str
        numeric = utils.list_columns_with_semantic_types(
            data.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        numeric = [x for x in numeric if x in all_attributes]
        for element in numeric:
            if data.metadata.query((mbase.ALL_ELEMENTS, element)).get('structural_type', ())==str:
                if pd.isnull(pd.to_numeric(data.iloc[:,element])).sum() == data.shape[0]:
                    self._empty_columns.append(element)

        # Remove columns with all empty values, structural numeric
        is_empty = pd.isnull(data).sum(axis=0) == data.shape[0]
        for i in all_attributes:
            if is_empty.iloc[i]:
                self._empty_columns.append(i)

        _logger.debug('Removing entirely empty columns: {}'.format(data.columns[self._empty_columns]))

        data = utils.remove_columns(data, self._empty_columns, source='ISI DSBox Data Encoder')

        categorical_attributes = utils.list_columns_with_semantic_types(metadata=data.metadata,
                                                                        semantic_types=[
                                                                            "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                                                                            "https://metadata.datadrivendiscovery.org/types/CategoricalData"])
        all_attributes = utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/Attribute"])

        self._cat_col_index = list(set(all_attributes).intersection(categorical_attributes))
        self._cat_columns = data.columns[self._cat_col_index].tolist()

        _logger.debug('Encoding columns: {}'.format(self._cat_columns))

        mapping = {}
        for column_name in self._cat_columns:
            col = data[column_name]
            temp = self._trim_features(col, self.hyperparams['n_limit'])
            if temp:
                mapping[temp[0]] = temp[1]
        self._mapping = mapping
        self._fitted = True

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        Convert and output the input data into encoded format,
        using the trained (fitted) encoder.
        Notice that [colname]_other_ and [colname]_nan columns
        are always kept for one-hot encoded columns.
        """

        self._input_data_copy = inputs.copy()

        # Remove columns with all empty values
        _logger.debug('Removing entirely empty columns: {}'.format(self._input_data_copy.columns[self._empty_columns]))
        self._input_data_copy = utils.remove_columns(self._input_data_copy, self._empty_columns, source='ISI DSBox Data Encoder')

        # Return if there is nothing to encode
        if len(self._cat_columns)==0:
            return CallResult(self._input_data_copy, True, 1)

        _logger.debug('Encoding columns: {}'.format(self._cat_columns))

        data_encode = self._input_data_copy[list(self._mapping.keys())]

        # Get rid of false SettingWithCopyWarning
        data_encode.is_copy = None

        res = []
        for column_name in self._cat_columns:
            feature = data_encode[column_name].copy()
            other_ = lambda x: 'Other' if (x and x not in self._mapping[column_name]) else x
            nan_ = lambda x: x if x else np.nan
            feature.loc[feature.notnull()] = feature[feature.notnull()].apply(other_)
            feature = feature.apply(nan_)
            new_column_names = ['{}_{}'.format(column_name, i) for i in self._mapping[column_name] + ['nan']]
            encoded = pd.get_dummies(feature, dummy_na=True, prefix=column_name)
            missed = [name for name in new_column_names if name not in list(encoded.columns)]
            for m in missed:
                # print('missing', m)
                encoded[m] = 0
            encoded = encoded[new_column_names]
            res.append(encoded)
            # data_encode.loc[:,column_name] = feature

        # Drop columns that will be encoded
        # data_rest = self._input_data_copy.drop(self._mapping.keys(), axis=1)
        columns_names = self._input_data_copy.columns.tolist()
        drop_indices = [columns_names.index(col)  for col in self._mapping.keys()]
        drop_indices = sorted(drop_indices)

        all_categorical = False
        try:
            self._input_data_copy = utils.remove_columns(self._input_data_copy, drop_indices,
                                                         source='ISI DSBox Data Encoder')
        except ValueError:
            _logger.warning("[warn] All the attributes are categorical!")
            all_categorical = True

        # metadata for columns that are not one hot encoded
        # self._col_index = [self._input_data_copy.columns.get_loc(c) for c in data_rest.columns]
        # data_rest.metadata = utils.select_columns_metadata(self._input_data_copy.metadata, self._col_index)

        # encode data
        # encoded = d3m_DataFrame(pd.get_dummies(data_encode, dummy_na=True, prefix=self._cat_columns, prefix_sep='_',
        #                                        columns=self._cat_columns))
        encoded = d3m_DataFrame(pd.concat(res, axis=1))

        # update metadata for existing columns

        for index in range(len(encoded.columns)):
            old_metadata = dict(encoded.metadata.query((mbase.ALL_ELEMENTS, index)))
            old_metadata["structural_type"] = int
            old_metadata["semantic_types"] = (
                'http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
            encoded.metadata = encoded.metadata.update((mbase.ALL_ELEMENTS, index), old_metadata)
        ## merge/concat both the dataframes
        if not all_categorical:
            output = utils.horizontal_concat(self._input_data_copy, encoded)
        else:
            output = encoded
        return CallResult(output, True, 1)

    def get_params(self) -> EncParams:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return EncParams(
            mapping=self._mapping,
            cat_columns=self._cat_columns,
            empty_columns=self._empty_columns)

    def set_params(self, *, params: EncParams) -> None:
        self._fitted = True
        self._mapping = params['mapping']
        self._cat_columns = params['cat_columns']
        self._empty_columns = params['empty_columns']

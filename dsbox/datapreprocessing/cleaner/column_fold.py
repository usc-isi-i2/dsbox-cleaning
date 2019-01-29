import pandas as pd
import dateparser
import numpy as np
from d3m import container
import d3m.metadata.base as mbase
from common_primitives import utils
from d3m.container import DataFrame as d3m_DataFrame
from dsbox.datapreprocessing.cleaner.dependencies.helper_funcs import HelperFunction

from typing import Dict

from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from . import config

# add more later if needed

special_characters = ['-', '_', ' ']

Input = container.DataFrame
Output = container.DataFrame


class FoldParams(params.Params):
    mapping: Dict


class FoldHyperparameter(hyperparams.Hyperparams):
    pass


class FoldColumns(UnsupervisedLearnerPrimitiveBase[Input, Output, FoldParams, FoldHyperparameter]):
    # TODO update metadata
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "dsbox-fold-columns",
        "version": config.VERSION,
        "name": "DSBox Fold Columns",
        "description": "Fold Columns",
        "python_path": "d3m.primitives.dsbox.FoldColumns",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": ["fold"],
        "installation": [config.INSTALLATION],
        "location_uris": [],
        "hyperparms_to_tune": []
    })

    def __repr__(self):
        return "%s(%r)" % ('Column_folder', self.__dict__)

    def __init__(self, *, hyperparams: FoldHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        self._df: Input = None
        self._ignore_list = list()
        self._mapping: Dict = {}
        self._fitted = False

    def set_training_data(self, *, inputs: Input) -> None:
        self._df = inputs
        self._fitted = False

    def get_params(self) -> FoldParams:
        if not self._fitted:
            raise ValueError("FoldColumns: Fit not performed.")
        return FoldParams(mapping=self._mapping)

    def set_params(self, *, params: FoldParams) -> None:
        self._fitted = True
        self._mapping = params['mapping']

    @staticmethod
    def _has_numbers(str):
        return any(char.isdigit() for char in str)

    @staticmethod
    def _has_numbers_or_special_characters(str):
        return any(char.isdigit() or char in special_characters for char in str)

    def _create_prefix_dict(self, column_names):
        """
        This function will create a dictionary with varying length prefixes of all column names
        :param column_names: the list of input column names
        :return: a dict: ['a','bc'] -> {'a':[0], 'b': [1], 'bc':[1]}
        """
        prefix_dict = {}
        for c in range(len(column_names)):
            column_name = column_names[c]
            # only take into consideration those columns which have a special character or numbers in it
            if self._has_numbers_or_special_characters(column_name):
                for i in range(len(column_name) + 1):
                    prefix = column_name[0:i]
                    if prefix != '' and not self._has_numbers(prefix) and len(prefix) > 1:
                        if column_name[0:i] not in prefix_dict:
                            prefix_dict[column_name[0:i]] = []
                        prefix_dict[column_name[0:i]].append(c)

        for key in list(prefix_dict):
            if len(prefix_dict[key]) <= 1:
                del prefix_dict[key]
        return prefix_dict

    def _check_if_seen(self, str1, seen_prefix_dict):
        """
        This function returns true if str1 is a subtring of any string in str_list and that string startswith str1
        :param str1: string to be checked
        :param str_lst: input list of strings
        :return: True if str1 is a substring of any of the strings else False
        """
        column_indices_seen = []
        for str in list(seen_prefix_dict):
            if str.startswith(str1):
                column_indices_seen.extend(seen_prefix_dict[str])

        if len(column_indices_seen) == 0:
            return False

        if len(list(set(self._prefix_dict[str1]) - set(column_indices_seen))) > 0:
            self._prefix_dict[str1] = list(set(self._prefix_dict[str1]) - set(column_indices_seen))
            return False
        return True

    @staticmethod
    def _check_if_numbers_contiguous(indices_list):
        for i in range(len(indices_list) - 1):
            if indices_list[i] != (indices_list[i + 1] - 1):
                return False
        return True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        call both prefix and date method and return  a combined list
        :return: list of list of columns to fold
        """
        if self._fitted:
            return

        if self._df is None:
            raise ValueError('Missing training(fitting) data.')

        self._column_names = list(self._df) if self._df is not None else []
        self._prefix_dict = self._create_prefix_dict(self._column_names)
        prefix_lst = self._detect_columns_to_fold_prefix()
        dates_list = self._detect_columns_to_fold_dates()
        c_list = []
        if len(prefix_lst) > 0:
            c_list.extend(prefix_lst)

        if len(dates_list) > 0:
            c_list.extend(dates_list)
        ret_lst = []
        for fold_group in c_list:
            semantic_types = set()
            for col in fold_group:
                old_metadata = dict(self._df.metadata.query((mbase.ALL_ELEMENTS, col)))
                semantic_types = semantic_types.union([x for x in old_metadata["semantic_types"] if x != "https://metadata.datadrivendiscovery.org/types/Attribute"])
            if len(semantic_types) <= 1:
                ret_lst.append(fold_group)

        self._mapping = {'foldable_columns': ret_lst}
        self._fitted = True
        return CallResult(None, has_finished=True, iterations_done=1)

    def _detect_columns_to_fold_prefix(self):
        sorted_prefix_lst = sorted(self._prefix_dict, key=len, reverse=True)
        valid_seen_prefixes = dict()

        for prefix in sorted_prefix_lst:
            if not self._check_if_seen(prefix, valid_seen_prefixes):
                if self._check_if_numbers_contiguous(self._prefix_dict[prefix]):
                    valid_seen_prefixes[prefix] = self._prefix_dict[prefix]

        for p in list(self._prefix_dict):
            if p not in valid_seen_prefixes:
                del self._prefix_dict[p]

        columns_list = []
        for key in list(self._prefix_dict):
            columns_list.extend(self._prefix_dict[key])

        return [self._prefix_dict[key] for key in list(self._prefix_dict)]

    def _detect_columns_to_fold_dates(self):
        """
        This function will return a list of column names if they are dates
        :return: a subset of column_names
        """
        result = list()
        for index in range(len(self._column_names)):
            column_name = self._column_names[index]
            # do not want 12 to be parsed as date, minimum length should be 4 (year in YYYY format)
            if len(column_name) >= 4:
                try:
                    # for now strict parsing is true, otherwise it'll parse 'year' as valid date.
                    # in future, we'll have to specify date formats
                    parsed_column_as_date = dateparser.parse(column_name, settings={'STRICT_PARSING': True})
                    if parsed_column_as_date:
                        # column_name has been parsed as a valid date, it is a candidate for fold
                        result.append(index)
                except:
                    # something went wrong, doesn't matter what
                    pass
        return result

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        columns_list_to_fold = self._mapping.get('foldable_columns', [])
        if len(columns_list_to_fold) == 0:
            return CallResult(inputs, True, 1)
        if inputs.shape[0] > 20000:
            return CallResult(inputs, True, 1)
        self._column_names = list(inputs) if inputs is not None else []
        df = None
        for columns_to_fold in columns_list_to_fold:
            df = self._fold_columns(inputs, columns_to_fold)
        cols_to_drop = list()
        for col_idx, col_name in enumerate(inputs.columns):
            if col_name not in df.columns:
                cols_to_drop.append(col_idx)

        inputs = utils.remove_columns(inputs, cols_to_drop)
        new_df = inputs[0:0]
        for col_name in new_df.columns:
            new_df.loc[:, col_name] = df.loc[:, col_name]

        extends = {}
        for col_name in df.columns:
            if col_name not in new_df.columns:
                extends[col_name] = df.loc[:, col_name].tolist()

        if extends:
            extends_df = d3m_DataFrame.from_dict(extends)
            extends_df.index = new_df.index.copy()
            new_df = utils.append_columns(new_df, extends_df)
            new_df = self._update_type(new_df, list(extends.keys()))

        old_metadata = dict(new_df.metadata.query(()))
        old_metadata["dimension"] = dict(old_metadata["dimension"])
        old_metadata["dimension"]["length"] = new_df.shape[0]
        new_df.metadata = new_df.metadata.update((), old_metadata)

        return CallResult(new_df, True, 1) if new_df is not None else CallResult(inputs, True, 1)

    def _fold_columns(self, inputs_df, columns_to_fold_all):
        columns_to_fold = list(set(columns_to_fold_all) - set(self._ignore_list))
        if len(columns_to_fold) == 0:
            # nothing to fold, return the original
            return inputs_df
        new_column_suffix = ''
        for c in columns_to_fold:
            new_column_suffix += '_' + str(c)
        new_column_name = '{}_{}'.format(inputs_df.columns[columns_to_fold[0]], new_column_suffix)
        new_rows_list = list()
        orig_columns = list(range(len(self._column_names)))
        # subtract ignore list from columns_to_fold

        non_foldable_columns = list(set(orig_columns) - set(columns_to_fold))

        for i in range(inputs_df.shape[0]):
            row = inputs_df.iloc[i]
            for column_to_fold in columns_to_fold:

                d1 = {}
                for nfc in non_foldable_columns:
                    d1[self._column_names[nfc]] = row[self._column_names[nfc]]

                d1[new_column_name] = self._column_names[column_to_fold]
                d1['{}_value'.format(new_column_name)] = row[column_to_fold]

                # record d3mIndex version. If you want using pandas default, comment out this block and uncomment next block
        #         d1['d3mIndex_reference'] = inputs_df.index[i]
        #         new_rows_list.append(d1)
        #
        # new_df = pd.DataFrame(new_rows_list).set_index('d3mIndex_reference', drop=True)
        # new_df.index.names = ['d3mIndex']

        # Not record d3mIndex. using pandas default
                new_rows_list.append(d1)

        new_df = pd.DataFrame(new_rows_list)

        return new_df

    @staticmethod
    def _update_type(df, added_cols):

        indices = list()
        for key in added_cols:
            indices.append(df.columns.get_loc(key))

        for idx in indices:
            old_metadata = dict(df.metadata.query((mbase.ALL_ELEMENTS, idx)))

            numerics = pd.to_numeric(df.iloc[:, idx], errors='coerce')
            length = numerics.shape[0]
            nans = numerics.isnull().sum()

            if nans / length > 0.9:
                if HelperFunction.is_categorical(df.iloc[:, idx]):
                    old_metadata['semantic_types'] = (
                        "https://metadata.datadrivendiscovery.org/types/CategoricalData",)
                else:
                    old_metadata['semantic_types'] = ("http://schema.org/Text",)
                    old_metadata['structural_type'] = type("type")
            else:
                intcheck = (numerics % 1) == 0
                if np.sum(intcheck) / length > 0.9:
                    old_metadata['semantic_types'] = ("http://schema.org/Integer",)
                    old_metadata['structural_type'] = type(10)
                else:
                    old_metadata['semantic_types'] = ("http://schema.org/Float",)
                    old_metadata['structural_type'] = type(10.1)

            old_metadata['semantic_types'] += ("https://metadata.datadrivendiscovery.org/types/Attribute",)

            df.metadata = df.metadata.update((mbase.ALL_ELEMENTS, idx), old_metadata)

        return df

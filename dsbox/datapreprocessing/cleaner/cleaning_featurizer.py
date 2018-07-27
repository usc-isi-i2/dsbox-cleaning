import logging
import pandas as pd
import numpy as np

from d3m import container
import d3m.metadata.base as mbase
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import Dict, Union

from common_primitives import utils
from dsbox.datapreprocessing.cleaner.dependencies.date_featurizer_org import DateFeaturizerOrg
from dsbox.datapreprocessing.cleaner.dependencies.spliter import PhoneParser, PunctuationParser, NumAlphaParser
from dsbox.datapreprocessing.cleaner.dependencies.helper_funcs import HelperFunction

from . import config

_logger = logging.getLogger(__name__)

Input = container.DataFrame
Output = container.DataFrame

Sample_rows = 100

Clean_operations = {
    "split_date_column": True,
    "split_phone_number_column": True,
    "split_alpha_numeric_column": True,
    "split_punctuation_column": True
}


class CleaningFeaturizerParams(params.Params):
    mapping: Dict


class CleaningFeaturizerHyperparameter(hyperparams.Hyperparams):
    features = hyperparams.Hyperparameter[Union[str, None]](
        None,
        description = 'Select one or more operations to perform: "split_date_column", "split_phone_number_column", "split_alpha_numeric_column", "split_multi_value_column"',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    num_threshold = hyperparams.Uniform(
        default = 0.1,
        lower = 0.1,
        upper = 0.5,
        upper_inclusive = True,
        description = 'Threshold for number character density of a column',
        semantic_types = ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    common_threshold = hyperparams.Uniform(
        default = 0.9,
        lower = 0.7,
        upper = 0.9,
        upper_inclusive = True,
        description = 'Threshold for rows containing specific punctuation',
        semantic_types = ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/ControlParameter'])

# class Params(params.Params):
#     components_: typing.Any


class CleaningFeaturizer(UnsupervisedLearnerPrimitiveBase[Input, Output, CleaningFeaturizerParams, CleaningFeaturizerHyperparameter]):
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "dsbox-cleaning-featurizer",
        "version": config.VERSION,
        "name": "DSBox Cleaning Featurizer",
        "description": "Split single column into multile columns based on the semantics of the column. The semantics this primitive can detect include: phone numbers, dates, alpha numeric values, and multi-value columns",
        "python_path": "d3m.primitives.dsbox.CleaningFeaturizer",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": [ "DATA_CONVERSION" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [ config.REPOSITORY ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "cleaning", "date", "phone", "alphanumeric", "punctuation" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "hyperparms_to_tune": []
        })

    def __repr__(self):
        return "%s(%r)" % ('Cleaner', self.__dict__)

    def __init__(self, *, hyperparams: CleaningFeaturizerHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._mapping: Dict = {}
        self._input_data: Input = None
        self._input_data_copy = None
        self._fitted = False
        self._col_index = None

        self._clean_operations = Clean_operations

    def get_params(self) -> CleaningFeaturizerParams:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return CleaningFeaturizerParams(
            mapping=self._mapping)

    def set_params(self, *, params: CleaningFeaturizerParams) -> None:
        self._fitted = True
        self._mapping = params['mapping']

    def set_training_data(self, *, inputs: Input) -> None:
        self._input_data = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self._fitted:
            return

        if self._input_data is None:
            raise ValueError('Missing training(fitting) data.')

        if self._clean_operations:
            data = self._input_data.copy()
            mapping = dict()

            if self._clean_operations.get("split_date_column"):
                date_cols = self._get_date_cols(data)
                if date_cols:
                    mapping["date_columns"] = date_cols

            if self._clean_operations.get("split_phone_number_column"):
                phone_cols = self._get_phone_cols(data, ignore_list=mapping.get("date_columns", []))
                if phone_cols.get("columns_to_perform"):
                    mapping["phone_columns"] = phone_cols

            if self._clean_operations.get("split_alpha_numeric_column"):
                alpha_numeric_cols = self._get_alpha_numeric_cols(data,
                                                                  ignore_list=mapping.get("date_columns", []) +
                                                                  mapping.get("phone_columns", {}).get("columns_to_perform", []))
                if alpha_numeric_cols.get("columns_to_perform"):
                    mapping["alpha_numeric_columns"] = alpha_numeric_cols

            if self._clean_operations.get("split_punctuation_column"):
                punctuation_cols = self._get_punctuation_cols(data,
                                                              ignore_list=mapping.get("date_columns", []) +
                                                              mapping.get("phone_columns", {}).get("columns_to_perform", [])                                                              )
                if punctuation_cols.get("columns_to_perform"):
                    mapping["punctuation_columns"] = punctuation_cols

            self._mapping = mapping

        self._fitted = True

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        self._input_data_copy = inputs.copy()
        cols_to_drop = list()

        date_cols = self._mapping.get("date_columns")
        if date_cols:
            cols_to_drop += self._mapping.get("date_columns")
            original_cols = self._get_cols(self._input_data_copy)
            dfo = DateFeaturizerOrg(dataframe=self._input_data_copy)
            df = dfo.featurize_date_columns(date_cols)
            current_cols = self._get_cols(df["df"])

            _logger.info(
                "Date Featurizer. 'created_columns': '%(created_columns)s'.",
                {
                    'created_columns': str(list(set(current_cols).difference(original_cols))),
                },
            )

            self._input_data_copy = df["df"]

        phone_cols = self._mapping.get("phone_columns")
        if phone_cols:
            cols_to_drop += phone_cols.get("columns_to_perform", [])
            original_cols = self._get_cols(self._input_data_copy)
            df = PhoneParser.perform(df=self._input_data_copy, columns_perform=phone_cols)
            current_cols = self._get_cols(df)

            _logger.info(
                "Phone Featurizer. 'created_columns': '%(created_columns)s'.",
                {
                    'created_columns': str(list(set(current_cols).difference(original_cols))),
                },
            )

            self._input_data_copy = df

        an_cols = self._mapping.get("alpha_numeric_columns")
        if an_cols:
            cols_to_drop += an_cols.get("columns_to_perform", [])
            original_cols = self._get_cols(self._input_data_copy)
            df = NumAlphaParser.perform(df=self._input_data_copy, columns_perform=an_cols)
            current_cols = self._get_cols(df)

            _logger.info(
                "NumAlpha Featurizer. 'created_columns': '%(created_columns)s'.",
                {
                    'created_columns': str(list(set(current_cols).difference(original_cols))),
                },
            )

            self._input_data_copy = df

        punc_cols = self._mapping.get("punctuation_columns")
        if punc_cols:
            cols_to_drop += punc_cols.get("columns_to_perform", [])
            original_cols = self._get_cols(self._input_data_copy)
            df = PunctuationParser.perform(df=self._input_data_copy, columns_perform=punc_cols)
            current_cols = self._get_cols(df)

            _logger.info(
                "Punctuation Featurizer. 'created_columns': '%(created_columns)s'.",
                {
                    'created_columns': str(list(set(current_cols).difference(original_cols))),
                },
            )

            self._input_data_copy = df

        if cols_to_drop:
            self._input_data_copy = utils.remove_columns(self._input_data_copy, cols_to_drop)
        self._update_structural_type()
        return CallResult(self._input_data_copy, True, 1)

    @staticmethod
    def _get_date_cols(data):
        dates = utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/Time"])

        return dates

    @staticmethod
    def _get_phone_cols(data, ignore_list):
        return PhoneParser.detect(df=data.iloc[:Sample_rows, :], columns_ignore=ignore_list)

    @staticmethod
    def _get_alpha_numeric_cols(data, ignore_list):
        return NumAlphaParser.detect(df=data.iloc[:Sample_rows, :], columns_ignore=ignore_list)

    @staticmethod
    def _get_punctuation_cols(data, ignore_list):
        return PunctuationParser.detect(df=data.iloc[:Sample_rows, :], columns_ignore=ignore_list)

    @staticmethod
    def _get_cols(df):
        return range(df.shape[1])

    def _update_structural_type(self):
        for col in range(self._input_data_copy.shape[1]):
            old_metadata = dict(self._input_data_copy.metadata.query((mbase.ALL_ELEMENTS, col)))
            semantic_type = old_metadata.get('semantic_types', None)
            if not semantic_type:
                numerics = pd.to_numeric(self._input_data_copy.iloc[:, col], errors='coerce')
                length = numerics.shape[0]
                nans = numerics.isnull().sum()

                if nans / length > 0.9:
                    if HelperFunction.is_categorical(self._input_data_copy.iloc[:, col]):
                        old_metadata['semantic_types'] = (
                        "https://metadata.datadrivendiscovery.org/types/CategoricalData",)
                    else:
                        old_metadata['semantic_types'] = ("http://schema.org/Text",)
                else:
                    intcheck = (numerics % 1) == 0
                    if np.sum(intcheck) / length > 0.9:
                        old_metadata['semantic_types'] = ("http://schema.org/Integer",)
                        old_metadata['structural_type'] = type(10)
                        self._input_data_copy.iloc[:, col] = numerics
                    else:
                        old_metadata['semantic_types'] = ("http://schema.org/Float",)
                        old_metadata['structural_type'] = type(10.2)
                        self._input_data_copy.iloc[:, col] = numerics

                old_metadata['semantic_types'] += ("https://metadata.datadrivendiscovery.org/types/Attribute",)

            else:
                if "http://schema.org/Integer" in semantic_type:
                    self._input_data_copy.iloc[:, col] = pd.to_numeric(self._input_data_copy.iloc[:, col],
                                                                       errors='coerce')
                    old_metadata['structural_type'] = type(10)
                elif "http://schema.org/Float" in semantic_type:
                    self._input_data_copy.iloc[:, col] = pd.to_numeric(self._input_data_copy.iloc[:, col],
                                                                       errors='coerce')
                    old_metadata['structural_type'] = type(10.2)

            self._input_data_copy.metadata = self._input_data_copy.metadata.update((mbase.ALL_ELEMENTS, col),
                                                                                   old_metadata)

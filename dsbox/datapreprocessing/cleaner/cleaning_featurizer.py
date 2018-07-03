
from d3m import container
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import NamedTuple, Dict, List, Set, Union

from common_primitives import utils
from dsbox.datapreprocessing.cleaner.date_featurizer_org import DateFeaturizerOrg
from dsbox.datapreprocessing.cleaner.spliter import PhoneParser, PunctuationParser, NumAlphaParser

from . import config

Input = container.DataFrame
Output = container.DataFrame

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
                phone_cols = self._get_phone_cols(data)
                if phone_cols:
                    mapping["phone_columns"] = phone_cols

            if self._clean_operations.get("split_alpha_numeric_column"):
                alpha_numeric_cols = self._get_alpha_numeric_cols(data)
                if alpha_numeric_cols:
                    mapping["alpha_numeric_columns"] = alpha_numeric_cols

            if self._clean_operations.get("split_punctuation_column"):
                punctuation_cols = self._get_punctuation_cols(data)
                if punctuation_cols:
                    mapping["punctuation_columns"] = punctuation_cols

            self._mapping = mapping

        self._fitted = True

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        self._input_data_copy = inputs.copy()
        if self._mapping.get("date_columns"):
            dfo = DateFeaturizerOrg(dataframe=self._input_data_copy)
            df = dfo.featurize_date_columns(self._mapping.get("date_columns"))
            self._input_data_copy = df["df"]

        if self._mapping.get("phone_columns"):
            ps = PhoneParser(df=self._input_data_copy)
            self._input_data_copy = ps.perform(self._mapping.get("phone_columns"))

        if self._mapping.get("alpha_numeric_columns"):
            nap = NumAlphaParser(df=self._input_data_copy)
            self._input_data_copy = nap.perform(self._mapping.get("alpha_numeric_columns"))

        if self._mapping.get("punctuation_columns"):
            ps = PunctuationParser(df=self._input_data_copy)
            self._input_data_copy = ps.perform(self._mapping.get("punctuation_columns"))

        return CallResult(self._input_data_copy, True, 1)

    @staticmethod
    def _get_date_cols(data):
        dates = set(utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/CategoricalData"])).intersection(
            utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/Time"]))

        return list(dates)

    @staticmethod
    def _get_phone_cols(data):
        return utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber"])

    @staticmethod
    def _get_alpha_numeric_cols(data):
        return utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/CanBeSplitByAlphanumeric"])

    @staticmethod
    def _get_punctuation_cols(data):
        return utils.list_columns_with_semantic_types(metadata=data.metadata, semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/CanBeSplitByPunctuation"])


from d3m import container
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
import time
from datetime import datetime
import pandas as pd
from warnings import warn
#from dependencies.date_extractor import DateExtractor
from . import config
import typing

Inputs = container.DataFrame
Outputs = container.DataFrame
clean_operations = ["split_phone_number_column", "split_date_column", "split_alpha_numeric_column", "split_multi_value_column"]

class DataFeaturizerHyperparameter(hyperparams.Hyperparams):
    create_year = hyperparams.UniformBool(
        default = True,
        description = 'define whether to create the year column or not',
        semantic_types=['http://schema.org/Boolean','https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    create_month = hyperparams.UniformBool(
        default = True,
        description = 'define whether to create the month column or not',
        semantic_types=['http://schema.org/Boolean','https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    create_day = hyperparams.UniformBool(
        default = True,
        description = 'define whether to create the day column or not',
        semantic_types=['http://schema.org/Boolean','https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    create_day_of_week = hyperparams.UniformBool(
        default = True,
        description = 'define whether to create the day of week column or not',
        semantic_types=['http://schema.org/Boolean','https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    min_threshold = hyperparams.Uniform(
        default = 0.9,
        lower = 0.0,
        upper = 1.0,
        upper_inclusive = True,
        description = 'Fraction of values required to be parsed as dates in order to featurize the column',
        semantic_types=['http://schema.org/Float','https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    extractor_settings = hyperparams.Hyperparameter[typing.Union[str, None]](
        None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="[Dict] Extractor settings for the date parser ",
    )


# class Params(params.Params):
#     pass

class DateFeaturizer(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, DataFeaturizerHyperparameter]):
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "dsbox-date-featurizer",
        "version": config.VERSION,
        "name": "DSBox Date Featurizer",
        "description": "Detect and trasform the date type data from the input",
        "python_path": "d3m.primitives.dsbox.DateFeaturizer",
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
        "keywords": [ "featurizer", "dates" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: DataFeaturizerHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

    # def set_params(self, *, params: Params) -> None:
    #     pass

    # def get_params(self) -> Params:
    #     pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return CallResult(inputs, True, True)


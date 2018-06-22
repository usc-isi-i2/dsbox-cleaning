
from d3m import container
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

from . import config

Inputs = container.DataFrame
Outputs = container.DataFrame

clean_operations = ["split_phone_number_column", "split_date_column", "split_alpha_numeric_column", "split_multi_value_column"]

class CleaningFeaturizerHyperparameter(hyperparams.Hyperparams):
    features = hyperparams.Set(
        clean_operations, clean_operations, 1, len(clean_operations),
        description = 'Select one or more operations to perform: "split_phone_number_column", "split_date_column", "split_alpha_numeric_column", "split_multi_value_column"',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    num_threshold = hyperparams.Hyperparameter[float](
        default = 0.1,
        lower = 0.1,
        upper = 0.5,
        description = 'Threshold for number character density of a column',
        semantic_types = ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    common_threshold = hyperparams.Hyperparameter[float](
        default = 0.9,
        lower = 0.7,
        upper = 0.9,
        description = 'Threshold for rows containing specific punctuation',
        semantic_types = ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/ControlParameter'])

class Params(params.Params):
    pass

class CleaningFeaturizer(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, CleaningFeaturizerHyperparameter]):
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "dsbox-cleaning-featurizer",
        "version": config.VERSION,
        "name": "DSBox Cleaning Featurizer",
        "description": "Splits single column into multile columns based on the semantics of the column. The semantics this primitive can detect include: phone numbers, dates, alpha numeric values, and multi-value columns",
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
        "keywords": [ "cleaning", "dates", "phones" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "hyperparms_to_tune": []
        })

    def __init__(self):
        pass

    def set_params(self, *, params: Params) -> None:
        pass

    def get_params(self) -> Params:
        pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return CallResult(inputs, True, True)

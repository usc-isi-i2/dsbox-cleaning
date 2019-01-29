import typing
import pandas as pd
import d3m.metadata.base as mbase
from . import config
from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from d3m import container
from numpy import ndarray
from common_primitives import utils
from sklearn.preprocessing import RobustScaler
from d3m.metadata import hyperparams, params
from d3m.container import DataFrame as d3m_DataFrame
from d3m.primitive_interfaces.base import CallResult
import numpy as np

__all__ = ('IQRScaler',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    center: Optional[ndarray]
    scale: Optional[ndarray]
    s_cols: typing.Iterable[typing.Any]
    fitted: typing.Union[bool, None]


class IQRHyperparams(hyperparams.Hyperparams):
    quantile_range_lowerbound = hyperparams.Uniform(
        lower=0.0,
        upper=25.0,
        default=25.0,
        upper_inclusive=True,
        description="IQR - Quantile range used to calculate scale",
        semantic_types=["http://schema.org/Float",
                        "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    quantile_range_upperbound = hyperparams.Uniform(
        lower=75.0,
        upper=100.0,
        default=75.0,
        upper_inclusive=True,
        description="IQR - Quantile range used to calculate scale",
        semantic_types=["http://schema.org/Float",
                        "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    with_centering = hyperparams.UniformBool(
        default=True,
        description=" If True, center the data before scaling ",
        semantic_types=["http://schema.org/Boolean",
                        "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    with_scaling = hyperparams.UniformBool(
        default=True,
        description="If True, scale the data to unit variance "
                    "(or equivalently, unit standard deviation).",
        semantic_types=["http://schema.org/Boolean",
                        "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


class IQRScaler(FeaturizationLearnerPrimitiveBase[Inputs, Outputs, Params, IQRHyperparams]):
    """
        A primitive which scales all the Integer & Float variables in the Dataframe.
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-multi-table-feature-scaler",
        "version": config.VERSION,
        "name": "DSBox feature scaler",
        "description": "A simple primitive that scales all the Integer & Float features with "
                       "sklearn's robust scaler",
        "python_path": "d3m.primitives.dsbox.IQRScaler",
        "primitive_family": "NORMALIZATION",
        "algorithm_types": ["DATA_NORMALIZATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["NORMALIZATION", "Scaler"],
        "installation": [config.INSTALLATION],
        "precondition": ["NO_MISSING_VALUES", "NO_CATEGORICAL_VALUES"],
        "effects": ["NO_JAGGED_VALUES"],
        "hyperparms_to_tune": [" inter quantile range"]
    })

    def __init__(self, *, hyperparams: IQRHyperparams) -> None:
        super(IQRScaler,self).__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._model = RobustScaler(with_centering=self.hyperparams['with_centering'],
                                   with_scaling=self.hyperparams['with_scaling'],
                                   quantile_range=(self.hyperparams['quantile_range_lowerbound'],
                                                   self.hyperparams['quantile_range_upperbound']))

        self._training_data = None
        self._fitted = False
        self._s_cols = None

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_data = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        numerical_attributes = utils.list_columns_with_semantic_types(
            metadata=self._training_data.metadata,
            semantic_types=["http://schema.org/Float", "http://schema.org/Integer"])

        all_attributes = utils.list_columns_with_semantic_types(
            metadata=self._training_data.metadata,
            semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"])
        self._s_cols = list(set(all_attributes).intersection(numerical_attributes))
        # print(" %d columns scaled" % (len(self._s_cols)))
        if len(self._s_cols) > 0:
            self._model.fit(self._training_data.iloc[:, self._s_cols])
            self._fitted = True
        else:
            self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) \
            -> CallResult[Outputs]:
        if not self._fitted:
            return CallResult(inputs, True, 1)
        # If `inputs` has index, then this statement will cause the for loop to introduce blank rows
        # temp = pd.DataFrame(self._model.transform(inputs.iloc[:, self._s_cols]))
        temp = pd.DataFrame(self._model.transform(inputs.iloc[:, self._s_cols]), index=inputs.index)
        outputs = inputs.copy()
        for id_index, od_index in zip(self._s_cols, range(temp.shape[1])):
            outputs.iloc[:, id_index] = temp.iloc[:, od_index]

        new_dtype = temp.dtypes
        lookup = {
            "float": ('http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/Attribute'),
            "int": ('http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute')
        }

        for d, index in zip(new_dtype, self._s_cols):
            # print("old metadata : ", outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            old_metadata = dict(outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            if d == np.dtype(np.float16) or d == np.dtype(np.float32) or \
                    d == np.dtype(np.float64) or d == np.dtype(np.float128):
                old_metadata["semantic_types"] = lookup["float"]
                old_metadata["structural_type"] = type(10.0)
            else:
                old_metadata["semantic_types"] = lookup["int"]
                old_metadata["structural_type"] = type(10)
            outputs.metadata = outputs.metadata.update((mbase.ALL_ELEMENTS, index), old_metadata)
            # print("updated dict : ",old_metadata)
            # print("check again : ", outputs.metadata.query((mbase.ALL_ELEMENTS, index)))

        if outputs.shape == inputs.shape:
            return CallResult(d3m_DataFrame(outputs), True, 1)
        else:
            return CallResult(inputs, True, 1)

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: IQRHyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce


    @classmethod
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int, hyperparams: IQRHyperparams) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    def get_params(self) -> Params:
        return Params(
            fitted = self._fitted,
            center = getattr(self._model, 'center_', None),
            scale = getattr(self._model, 'scale_', None),
            s_cols = self._s_cols
        )

    def set_params(self, *, params: Params) -> None:
        self._model.center_ = params['center']
        self._model.scale_ = params['scale']
        self._s_cols = params['s_cols']
        self._fitted = params['fitted']

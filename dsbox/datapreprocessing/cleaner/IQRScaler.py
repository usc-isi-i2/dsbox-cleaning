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
    center_: Optional[ndarray]
    scale_: Optional[ndarray]
    s_cols: List[object]


class IQRHyperparams(hyperparams.Hyperparams):
    quantile_range_lowerbound = hyperparams.Uniform(
        lower=0.0,
        upper=25.0,
        default=25.0,
        upper_inclusive=True,
        description="IQR - Quantile range used to calculate scale",
        semantic_types=["http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    quantile_range_upperbound = hyperparams.Uniform(
        lower=75.0,
        upper=100.0,
        default=75.0,
        upper_inclusive=True,
        description="IQR - Quantile range used to calculate scale",
        semantic_types=["http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    with_centering = hyperparams.UniformBool(
        default=True,
        description=" If True, center the data before scaling ",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    with_scaling = hyperparams.UniformBool(
        default=True,
        description="If True, scale the data to unit variance (or equivalently, unit standard deviation).",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
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
        "description": "A simple primitive that scales all the Integer & Float features with sklearn's robust scaler",
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
        super().__init__(hyperparams=hyperparams)
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
        numerical_attributes = utils.list_columns_with_semantic_types(metadata=self._training_data.metadata,
                                                                               semantic_types=[
                                                                                   "http://schema.org/Float",
                                                                                   "http://schema.org/Integer"])

        all_attributes = utils.list_columns_with_semantic_types(metadata=self._training_data.metadata,
                                                                         semantic_types=[
                                                                             "https://metadata.datadrivendiscovery.org/types/Attribute"])
        self._s_cols = list(set(all_attributes).intersection(numerical_attributes))
        print(" %d columns scaled" % (len(self._s_cols)))
        if len(self._s_cols) > 0:
            self._model.fit(self._training_data.iloc[:, self._s_cols])
            self._fitted = True
        else:
            self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            return CallResult(inputs, True, 1)
        temp = pd.DataFrame(self._model.transform(inputs.iloc[:, self._s_cols]))
        outputs = inputs.copy()
        for id_index, od_index in zip(self._s_cols, range(temp.shape[1])):
            outputs.iloc[:, id_index] = temp.iloc[:, od_index]

        new_dtype = temp.dtypes
        lookup = {"float": ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'),
                  "int": ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')}

        for d, index in zip(new_dtype, self._s_cols):
            print("old metadata : ", outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            old_metadata = dict(outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            if d==np.dtype(np.float16) or d==np.dtype(np.float32) or d==np.dtype(np.float64) or d==np.dtype(np.float128):
                old_metadata["semantic_types"] = lookup["float"]
                old_metadata["structural_type"] = type(10.0)
            else:
                old_metadata["semantic_types"] = lookup["int"]
                old_metadata["structural_type"] = type(10)
            outputs.metadata = outputs.metadata.update((mbase.ALL_ELEMENTS, index),old_metadata)
            print("updated dict : ",old_metadata)
            print("check again : ", outputs.metadata.query((mbase.ALL_ELEMENTS, index)))

        if outputs.shape == inputs.shape:
            return CallResult(d3m_DataFrame(outputs), True, 1)
        else:
            return CallResult(inputs, True, 1)

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(
            coef_=getattr(self._model, 'center_', None),
            intercept_=getattr(self._model, 'scale_', None),
            s_cols = self._s_cols
            )

    def set_params(self, *, params: Params) -> None:
        self._model.center_ = params['center_']
        self._model.scale_ = params['scale_']
        self._s_cols = params['s_cols']
        self._fitted = True

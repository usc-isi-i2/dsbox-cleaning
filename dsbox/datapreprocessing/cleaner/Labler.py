import pandas as pd
import numpy as np
from numpy import ndarray
import typing
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from d3m import container
import d3m.metadata.base as mbase
from . import config
from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
from common_primitives import utils
from d3m.metadata import hyperparams, params
from d3m.container import DataFrame as d3m_DataFrame
from d3m.primitive_interfaces.base import CallResult
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

__all__ = ('Labler',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    classes_: Optional[ndarray]


class Hyperparams(hyperparams.Hyperparams):
    pass


class Labler(FeaturizationLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        A primitive which scales all the Integer & Float variables in the Dataframe.
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-multi-table-feature-labler",
        "version": config.VERSION,
        "name": "DSBox feature labeler",
        "description": "A simple primitive that labels all string based categorical columns",
        "python_path": "d3m.primitives.dsbox.Labler",
        "primitive_family": "NORMALIZATION",
        "algorithm_types": ["DATA_NORMALIZATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["NORMALIZATION", "Labler"],
        "installation": [config.INSTALLATION],
        "precondition": ["NO_MISSING_VALUES", "CATEGORICAL_VALUES"],

    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        #self._model = LabelEncoder()
        #self._fits =[]
        self._training_data = None
        self._fitted = False
        self._s_cols = None
        self._temp = pd.DataFrame()
        self._d = defaultdict(LabelEncoder)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_data = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        categorical_attributes = utils.list_columns_with_semantic_types(metadata=self._training_data.metadata,
                                                                               semantic_types=[
                                                                                   "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                                                                                   "https://metadata.datadrivendiscovery.org/types/CategoricalData"
                                                                               ])

        all_attributes = utils.list_columns_with_semantic_types(metadata=self._training_data.metadata,
                                                                         semantic_types=[
                                                                             "https://metadata.datadrivendiscovery.org/types/Attribute"])

        self._s_cols = list(set(all_attributes).intersection(categorical_attributes))
        print(" %d of categorical attributes " % (len(self._s_cols)))

        if len(self._s_cols) > 0:
            self._training_data.iloc[:,self._s_cols].apply(lambda x: self._d[x.name].fit(x))
            self._fitted = True
        else:
            self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            return CallResult(inputs, True, 1)
        temp = pd.DataFrame(self._training_data.iloc[:, self._s_cols].apply(lambda x: self._d[x.name].transform(x)))
        outputs = self._training_data.copy()

        for id_index, od_index in zip(self._s_cols, range(temp.shape[1])):
            outputs.iloc[:, id_index] = temp.iloc[:, od_index]

        lookup = {"int": ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')}

        #new_dtype = temp.dtypes

        for index in self._s_cols:
            old_metadata = dict(outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            old_metadata["semantic_types"] = lookup["int"]
            old_metadata["structural_type"] = type(10)
            outputs.metadata = outputs.metadata.update((mbase.ALL_ELEMENTS, index), old_metadata)

        if outputs.shape == inputs.shape:
            print("output:",outputs.head(5))
            return CallResult(d3m_DataFrame(outputs), True, 1)
        else:
            return CallResult(inputs, True, 1)

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(
            classes_=getattr(self._model, 'classes_', None),
        )

    def set_params(self, *, params: Params) -> None:
        self._model.classes_s = params['classes_']
        self._fitted = True

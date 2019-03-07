import numpy as np #  type: ignore
import pandas as pd  #  type: ignore
from fancyimpute import KNN as knn  #  type: ignore

from . import missing_value_pred as mvp
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import stopit #  type: ignore
import math

from d3m import container
from d3m.metadata import hyperparams
from d3m.metadata.hyperparams import UniformInt, UniformBool, Hyperparams

from . import config

Input = container.DataFrame
Output = container.DataFrame

class KnnHyperparameter(Hyperparams):
    # A reasonable upper bound would the size of the input. For now using 100.
    k = UniformInt(lower=1, upper=100, default=5,
                   description='Number of neighbors',
                   semantic_types=['http://schema.org/Integer',
                                   'https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    verbose = UniformBool(default=False,
                          semantic_types=['http://schema.org/Boolean',
                                          'https://metadata.datadrivendiscovery.org/types/ControlParameter'])

class KNNImputation(TransformerPrimitiveBase[Input, Output, KnnHyperparameter]):
    """
    Impute the missing value using k nearest neighbors (weighted average).
    This class is a wrapper from fancyimpute-knn

    Parameters:
    ----------
    k: the number of nearest neighbors

    verbose: bool
        Control the verbosity

    """
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "faeeb725-6546-3f55-b80d-8b79d5ca270a",
        "version": config.VERSION,
        "name": "DSBox KNN Imputer",
        "description": "Impute missing values using k-nearest neighbor",
        "python_path": "d3m.primitives.data_cleaning.k_neighbors.DSBOX",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": [ "IMPUTATION", "K_NEAREST_NEIGHBORS" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [ config.REPOSITORY ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "preprocessing", "imputation", "knn" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "precondition": [ hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES ],
        # "effects": [ hyperparams.base.PrimitiveEffects.NO_MISSING_VALUES ],
        "hyperparms_to_tune": []
    })


    def __init__(self, *, hyperparams: KnnHyperparameter) -> None:

        super().__init__(hyperparams=hyperparams)
        # All primitives must define these attributes
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._train_x = None
        self._has_finished = False
        self._iterations_done = False
        self._verbose = hyperparams['verbose'] if hyperparams else False
        self._k = hyperparams['k'] if hyperparams else 5


    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        precond: run fit() before

        to complete the data, based on the learned parameters, support:
        -> greedy search

        also support the untrainable methods:
        -> iteratively regression
        -> other

        Parameters:
        ----------
        data: pandas dataframe
        label: pandas series, used for the evaluation of imputation

        TODO:
        ----------
        1. add evaluation part for __simpleImpute()

        """

        if (timeout is None):
            timeout = 2**31-1

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        else:
            data = inputs[0].copy()
        # record keys:
        keys = data.keys()
        index = data.index

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose: print("=========> impute by fancyimpute-knn:")
            data_clean = self.__knn(data)

        result = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._has_finished = True
            self._iterations_done = True
            result = pd.DataFrame(data_clean, index, keys)
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            self._has_finished = False
            self._iterations_done = False
        return CallResult(result, self._has_finished, self._iterations_done)


    #============================================ core function ============================================
    def __knn(self, test_data):
        """
        wrap fancyimpute-knn
        """
        missing_col_id = []
        test_data = mvp.df2np(test_data, missing_col_id, self._verbose)
        if (len(missing_col_id) == 0): return test_data
        complete_data = knn(k=self._k, verbose=(1 if self._verbose else 0)).complete(test_data)
        return complete_data

import numpy as np
import pandas as pd
from fancyimpute import MICE as mice

from . import missing_value_pred as mvp
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.base import CallResult
import stopit
import math
import typing

import d3m_metadata.container
from d3m_metadata import metadata
from d3m_metadata.metadata import PrimitiveMetadata
from d3m_metadata.hyperparams import UniformInt, Hyperparams
import collections

from . import config

Input = d3m_metadata.container.DataFrame
Output = d3m_metadata.container.DataFrame

class MiceHyperparameter(Hyperparams):
    verbose = UniformInt(lower=0, upper=1, default=0)

class MICE(TransformerPrimitiveBase[Input, Output, MiceHyperparameter]):
    """
    Impute the missing value using MICE. 
    This class is a wrapper from fancyimpute-mice

    Parameters:
    ----------
    verbose: Integer
        Control the verbosity
    """

    metadata = PrimitiveMetadata({
        ### Required
        "id": "3f72646a-6d70-3b65-ab42-f6a41552cecb",
        "version": config.VERSION, 
        "name": "DSBox MICE Imputer",
        "description": "Impute missing values using the MICE algorithm",   
        "python_path": "d3m.primitives.dsbox.MiceImputation",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": [ "IMPUTATION" ],
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
        "keywords": [ "preprocessing", "imputation" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "precondition": [ metadata.PrimitivePrecondition.NO_CATEGORICAL_VALUES ],
        "effects": [ metadata.PrimitiveEffects.NO_MISSING_VALUES ],
        "hyperparms_to_tune": []
    })


    def __init__(self, *, hyperparams: MiceHyperparameter, random_seed: int = 0, 
                 docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        # All primitives must define these attributes
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers

        # All other attributes must be private with leading underscore        
        self._train_x = None
        self._has_finished = False
        self._iterations_done = False
        self._verbose = hyperparams['verbose'] if hyperparams else 0


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
            timeout = math.inf
        if (iterations is None):
            iterations = 100   # default value for mice

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
            if (self._verbose>0): print("=========> impute by fancyimpute-mice:")
            data_clean = self.__mice(data, iterations)

        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._has_finished = True
            self._iterations_done = True
            value =pd.DataFrame(data_clean, index, keys)
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            self._has_finished = False
            self._iterations_done = False
        return CallResult(value, self._has_finished, self._iterations_done)



    #============================================ core function ============================================
    def __mice(self, test_data, iterations):
        """
        wrap fancyimpute-mice
        """
        missing_col_id = []
        test_data = mvp.df2np(test_data, missing_col_id, self._verbose)
        if (len(missing_col_id) == 0): return test_data
        complete_data = mice(n_imputations=iterations, verbose=self._verbose).complete(test_data)
        return complete_data


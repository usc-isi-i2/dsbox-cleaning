import numpy as np
import pandas as pd
from fancyimpute import MICE as mice

from . import missing_value_pred as mvp
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.base import CallMetadata
from typing import NamedTuple, Sequence
import stopit
import math

Input = pd.DataFrame
Output = pd.DataFrame


class MICE(TransformerPrimitiveBase[Input, Output]):
    """
    Impute the missing value using k nearest neighbors (weighted average). 
    This class is a wrapper from fancyimpute-mice

    Parameters:
    ----------
    verbose: Integer
        Control the verbosity
    """

    def __init__(self, verbose=0) -> None:
        self.train_x = None
        self.is_fitted = True
        self._has_finished = True
        self._iterations_done = True
        self.verbose = verbose


    def get_call_metadata(self) -> CallMetadata:
            return CallMetadata(has_finished=self._has_finished, iterations_done=self._iterations_done)



    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
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

        if (not self.is_fitted):
            # todo: specify a NotFittedError, like in sklearn
            raise ValueError("Calling produce before fitting.")

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
        
        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if (self.verbose>0): print("=========> impute by fancyimpute-mice:")
            data_clean = self.__mice(data, iterations)


        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self.is_fitted = True
            self._has_finished = True
            self._iterations_done = True
            return pd.DataFrame(data=data_clean, columns=keys)
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            self.is_fitted = False
            self._has_finished = False
            self._iterations_done = False
            return None



    #============================================ core function ============================================
    def __mice(self, test_data, iterations):
        """
        wrap fancyimpute-mice
        """
        test_data = mvp.df2np(test_data, [], self.verbose)
        complete_data = mice(n_imputations=iterations, verbose=self.verbose).complete(test_data)
        return complete_data


    # bellowing way is to combine the train_data and test_data, then do the mice imputation
    # but in usage, the user might input same data during through `set_training_data` and `produce`
    # therefore, for now let use not use the way
    # def __mice(self, test_data):
    #     """
    #     wrap fancyimpute-mice
    #     """
    #     test_data = mvp.df2np(test_data, [], self.verbose)
    #     break_point = test_data.shape[0]
    #     train_data = mvp.df2np(self.train_x, [], self.verbose)
    #     all_data = np.concatenate((test_data,train_data), axis=0)   # include more data to use
    #     complete_data = mice().complete(all_data)

    #     return complete_data[:break_point, :]



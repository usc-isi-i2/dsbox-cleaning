import numpy as np
import pandas as pd
from fancyimpute import SimpleFill

from . import missing_value_pred as mvp
# from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from primitive_interfaces.transformer import TransformerPrimitiveBase

from primitive_interfaces.base import CallMetadata
from typing import NamedTuple, Sequence
import stopit
import math

Input = pd.DataFrame
Output = pd.DataFrame

class MeanImputation(TransformerPrimitiveBase[Input, Output]):
    """
    Imputate the missing value using the `mean` value of the attribute
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
            if (self.verbose>0): print("=========> impute by mean value of the attribute:")
            data_clean = self.__mean(data)

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
    def __mean(self, test_data):
        """
        wrap fancyimpute-mean
        """
        test_data = mvp.df2np(test_data, [], self.verbose)
        complete_data = SimpleFill(fill_method="mean").complete(test_data)
        return complete_data

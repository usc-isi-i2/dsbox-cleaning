import numpy as np
import pandas as pd
from fancyimpute import SimpleFill

from . import missing_value_pred as mvp
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallMetadata
from typing import NamedTuple, Sequence
import stopit
import math

Input = pd.DataFrame
Output = pd.DataFrame

Params = NamedTuple("params", [
    ('verbose', int)]
    ) 


class MeanImputation(UnsupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    Imputate the missing value using the `mean` value of the attribute
    """

    def __init__(self) -> None:
        self.train_x = None
        self.is_fitted = False
        self._has_finished = False
        self.verbose = 0

    def set_params(self, verbose=0) -> None:
        self.verbose = verbose

    def get_params(self) -> Params:
        return Params(verbose=self.verbose)


    def get_call_metadata(self) -> CallMetadata:
            return CallMetadata(has_finished=self._has_finished, iterations_done=self._iterations_done)


    def set_training_data(self, *, inputs: Sequence[Input]) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs.
        """
        self.train_x = inputs
        self.is_fitted = False



    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        train imputation parameters. Now support:
        -> greedySearch

        for the method that not trainable, do nothing:
        -> interatively regression
        -> other

        Parameters:
        ----------
        data: pandas dataframe
        label: pandas series, used for the trainable methods
        """

        # if already fitted on current dataset, do nothing
        if self.is_fitted:
            return True

        # do noting in fit, no need to timeout
        self.is_fitted = True
        self._has_finished = True
        self._iterations_done = True

        # setup the timeout
        # with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
        #     assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING
 
        #     data = self.train_x.copy()
        #     label = self.train_y.copy()


        # if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
        #     self.is_fitted = True
        #     self._has_finished = True
        #     self._iterations_done = True
        # elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
        #     self.is_fitted = False
        #     self._has_finished = False
        #     self._iterations_done = False
        #     return


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

        data = inputs.copy()
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


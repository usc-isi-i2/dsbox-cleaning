import abc
from typing import *

from .base import *

__all__ = ('UnsupervisedLearnerPrimitiveBase',)


class UnsupervisedLearnerPrimitiveBase(PrimitiveBase[Input, Output, Params]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs from inputs, but they are fitted only on input data.
    """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Sequence[Input]) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs.
        """

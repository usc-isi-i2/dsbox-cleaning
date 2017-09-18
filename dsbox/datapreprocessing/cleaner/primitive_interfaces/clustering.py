from .base import *
from .unsupervised_learning import UnsupervisedLearnerPrimitiveBase

__all__ = ('ClusteringPrimitiveBase',)


class ClusteringPrimitiveBase(UnsupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    A base class for primitives implementing a clustering algorithm.
    """

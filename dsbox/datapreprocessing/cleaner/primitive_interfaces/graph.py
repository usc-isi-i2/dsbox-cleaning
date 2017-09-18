import abc
from typing import *

from .base import Output
from .transformer import TransformerPrimitiveBase
from .types.graph import Graph

__all__ = ('GraphTransformerPrimitiveBase',)

Input = TypeVar('Input', bound=Graph)


class GraphTransformerPrimitiveBase(TransformerPrimitiveBase[Input, Output]):
    """
    A base class for transformer primitives which take Graph objects as input.
    Graph is an interface which TA1 teams should implement for graph data.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
        """
        Produce primitive's best choice of the output for each of the inputs.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs of shape [num_inputs, ...].
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        Sequence[Output]
            The outputs of shape [num_inputs, ...].
        """

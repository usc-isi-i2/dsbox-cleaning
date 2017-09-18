from typing import *

from .base import *

__all__ = ('TransformerPrimitiveBase',)


class TransformerPrimitiveBase(PrimitiveBase[Input, Output, None]):
    """
    A base class for primitives which are not fitted at all and can
    simply produce (useful) outputs from inputs directly. As such they
    also do not have any state (params).

    This class is parametrized using only two type variables, ``Input`` and ``Output``.
    """

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        A noop.
        """

        return

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

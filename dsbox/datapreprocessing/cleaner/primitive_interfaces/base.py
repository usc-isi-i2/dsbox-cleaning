import abc
from typing import *
import random

__all__ = (
    'Input', 'Output', 'Params', 'PrimitiveBase', 'SamplingCompositionalityMixin',
    'ProbabilisticCompositionalityMixin', 'Scores', 'Gradients', 'GradientCompositionalityMixin',
    'InspectLossMixin',
)


Input = TypeVar('Input')
Output = TypeVar('Output')
# TODO: For now set without a bound because of a bug in mypy.
#       See: https://github.com/python/mypy/issues/3915
# Params = TypeVar('Params', bound=NamedTuple)
Params = TypeVar('Params')


# For description of values see ``get_call_metadata``.
CallMetadata = NamedTuple('CallMetadata', [
    ('has_finished', bool),
    ('iterations_done', Optional[int]),
])


class PrimitiveBase(Generic[Input, Output, Params]):
    """
    A base class for all TA1 primitives.

    Class is parametrized using three type variables, ``Input``, ``Output``, and ``Params``.
    ``Params`` has to be a subclass of a `NamedTuple` and subclasses of this class should
    define types for all fields of a provided named tuple.`

    All arguments to all methods are keyword-only. In Python 3 this is enforced, in Python 2
    this is not, but callers should still use only keyword-based arguments when calling to
    be backwards and future compatible.

    Subclasses of this class allow functional compositionality.
    """

    def __init__(self) -> None:
        """
        All primitives should specify all the hyper-parameters that can be set at the class
        level in their ``__init__`` as explicit typed keyword-only arguments
        (no ``*args`` or ``**kwargs``).

        Hyper-parameters are those primitive's parameters which are not changing during
        a life-time of a primitive. Parameters which do are set using the ``set_params`` method.
        """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
        """
        Produce primitive's best choice of the output for each of the inputs.

        In many cases producing an output is a quick operation in comparison with ``fit``, but not
        all cases are like that. For example, a primitive can start a potentially long optimization
        process to compute outputs. ``timeout`` and ``iterations`` can serve as a way for a caller
        to guide the length of this process.

        Ideally, a primitive should adapt its call to try to produce the best outputs possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before producing outputs, it should raise a ``TimeoutError`` exception to signal that the
        call was unsuccessful in the given time. The state of the primitive after the exception
        should be as the method call has never happened and primitive should continue to operate
        normally. The purpose of ``timeout`` is to give opportunity to a primitive to cleanly
        manage its state instead of interrupting execution from outside. Maintaining stable internal
        state should have precedence over respecting the ``timeout`` (caller can terminate the
        misbehaving primitive from outside anyway). If a longer ``timeout`` would produce
        different outputs, then ``get_call_metadata``'s ``has_finished`` should be set to
        ``False``.

        Some primitives have internal iterations (for example, optimization iterations).
        For those, caller can provide how many of primitive's internal iterations
        should a primitive do before returning outputs. Primitives should make iterations as
        small as reasonable. If ``iterations`` is ``None``, then there is no limit on
        how many iterations the primitive should do and primitive should choose the best amount
        of iterations on its own (potentially controlled through hyper-parameters).
        If ``iterations`` is a number, a primitive has to do those number of iterations,
        if possible. ``timeout`` should still be respected and potentially less iterations
        can be done because of that. Primitives with internal iterations should make
        ``get_call_metadata`` returns correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should run fully, respecting only ``timeout``.

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

    @abc.abstractmethod
    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        Fits primitive using inputs and outputs (if any) using currently set training data.

        If ``fit`` has already been called in the past on different training data,
        this method fits it **again from scratch** using currently set training data.

        On the other hand, caller can call ``fit`` multiple times on the same training data
        to continue fitting.

        If ``fit`` fully fits using provided training data, there is no point in making further
        calls to this method with same training data, and in fact further calls can be noops,
        or a primitive can decide to refit from scratch.

        In the case fitting can continue with same training data (even if it is maybe not reasonable,
        because the internal metric primitive is using looks like fitting will be degrading), if ``fit``
        is called again (without setting training data), the primitive has to continue fitting.

        Caller can provide ``timeout`` information to guide the length of the fitting process.
        Ideally, a primitive should adapt its fitting process to try to do the best fitting possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before fitting, it should raise a ``TimeoutError`` exception to signal that fitting was
        unsuccessful in the given time. The state of the primitive after the exception should be
        as the method call has never happened and primitive should continue to operate normally.
        The purpose of ``timeout`` is to give opportunity to a primitive to cleanly manage
        its state instead of interrupting execution from outside. Maintaining stable internal state
        should have precedence over respecting the ``timeout`` (caller can terminate the misbehaving
        primitive from outside anyway). If a longer ``timeout`` would produce different fitting,
        then ``get_call_metadata``'s ``has_finished`` should be set to ``False``.

        Some primitives have internal fitting iterations (for example, epochs). For those, caller
        can provide how many of primitive's internal iterations should a primitive do before returning.
        Primitives should make iterations as small as reasonable. If ``iterations`` is ``None``,
        then there is no limit on how many iterations the primitive should do and primitive should
        choose the best amount of iterations on its own (potentially controlled through
        hyper-parameters). If ``iterations`` is a number, a primitive has to do those number of
        iterations (even if not reasonable), if possible. ``timeout`` should still be respected
        and potentially less iterations can be done because of that. Primitives with internal
        iterations should make ``get_call_metadata`` returns correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should fit fully, respecting only ``timeout``.

        Subclasses can extend arguments of this method with explicit typed keyword arguments used during
        the fitting process. For example, they can accept other primitives through an argument representing
        a regularizer to use during fitting. The reason why those are not part of constructor arguments is
        that one can create primitives in any order before having to invoke them or pass them to other
        primitives.

        Parameters
        ----------
        timeout : float
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.
        """

    @abc.abstractmethod
    def get_params(self) -> Params:
        """
        Returns parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        Parameters should include all data which is necessary to create a new instance of this primitive
        behaving exactly the same as this instance, when the new instance is created by passing the same
        parameters to the class constructor and calling ``set_params``.

        Returns
        -------
        Params
            A named tuple of parameters.
        """

    @abc.abstractmethod
    def set_params(self, *, params: Params) -> None:
        """
        Sets parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """

    def get_call_metadata(self) -> CallMetadata:
        """
        Returns metadata about the last ``produce`` or ``fit`` call.

        For ``produce``, ``has_finished`` is ``True`` if the last call to ``produce``
        has produced the final outputs and a call with more time or more iterations
        cannot get different outputs.

        For ``fit``, ``has_finished`` is ``True`` if a primitive has been fully fitted
        on current training data and further calls to ``fit`` are unnecessary and will
        not change anything. ``False`` means that more iterations can be done (but it
        does not necessary mean that more iterations are beneficial).

        If a primitive has iterations internally, then ``iterations_done`` contains
        how many of those iterations have been made during the last call. If primitive
        does not support them, ``iterations_done`` is ``None``.

        The reason why this is a separate call is to make return value from ``produce`` and
        ``fit`` simpler. Moreover, not all callers might care about this information and for
        many primitives a default implementation of this method works.

        Returns
        -------
        CallMetadata
            A named tuple with metadata.
        """

        return CallMetadata(has_finished=True, iterations_done=None)

    def set_random_seed(self, *, seed: int) -> None:
        """
        Sets a random seed for all operations from now on inside the primitive.

        By default it sets numpy's and Python's random seed.

        Parameters
        ----------
        seed : int
            A random seed to use.
        """

        try:
            import numpy  # type: ignore
            numpy.random.seed(seed)
        except ImportError:
            pass

        random.seed(seed)


class ContinueFitMixin(Generic[Input, Output, Params]):
    @abc.abstractmethod
    def continue_fit(self, *, timeout: float = None, iterations: Optional[int] = 1) -> None:
        """
        Similar to base ``fit``, this method fits the primitive using inputs and outputs (if any)
        using currently set training data.

        The difference is what happens when currently set training data is different from
        what the primitive might have already been fitted on. ``fit`` fits the primitive from
        scratch, while ``continue_fit`` fits it further and does **not** start from scratch.

        Caller can still call ``continue_fit`` multiple times on the same training data as well,
        in which case primitive should try to improve the fit in the same way as with ``fit``.

        From the perspective of a caller of all other methods, the training data in effect
        is still just currently set training data. If a caller wants to call ``gradient_output``
        on all data on which the primitive has been fitted through multiple calls of ``continue_fit``
        on different training data, the caller should pass all this data themselves through
        another call to ``set_training_data``, do not call ``fit`` or ``continue_fit`` again,
        and use ``gradient_output`` method. In this way primitives which truly support
        continuation of fitting and need only the latest data to do another fitting, do not
        have to keep all past training data around themselves.

        If a primitive supports this mixin, then both ``fit`` and ``continue_fit`` can be
        called. ``continue_fit`` always continues fitting, if it was started through ``fit``
        or ``continue_fit``. And ``fit`` always restarts fitting, even if previously
        ``continue_fit`` was used.

        When this mixin is supported, then ``get_call_metadata`` method should return
        metadata also for call of ``continue_fit``.

        Parameters
        ----------
        timeout : float
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.
        """


class SamplingCompositionalityMixin(Generic[Input, Output, Params]):
    """
    This mixin signals to a caller that the primitive is probabilistic but
    may be likelihood free.

    Mixin should be used together with the ``PrimitiveBase`` class.
    """

    def sample_one(self, *, input: Input, num_samples: int = 1, timeout: float = None, iterations: Optional[int] = 1) -> Sequence[Output]:
        """
        Sample ``num_samples`` outputs for one input ``input``.

        Semantics of ``timeout`` and ``iterations`` is the same as in ``produce``.

        When this mixin is supported, then ``get_call_metadata`` method should return
        metadata also for call of ``sample_one``.

        Parameters
        ----------
        input : Input
            The input.
        num_samples : int
            The number of samples to return.
        timeout : float
            A maximum time this primitive should take to sample outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        Sequence[Output]
            The set of samples of shape [num_samples, ...].
        """

        return self.sample_multiple(inputs=[input], num_samples=num_samples, timeout=timeout, iterations=iterations)[0]

    @abc.abstractmethod
    def sample_multiple(self, *, inputs: Sequence[Input], num_samples: int = 1, timeout: float = None, iterations: Optional[int] = 1) -> Sequence[Sequence[Output]]:
        """
        Sample multiple inputs at once.

        Semantics of ``timeout`` and ``iterations`` is the same as in ``produce``.

        When this mixin is supported, then ``get_call_metadata`` method should return
        metadata also for call of ``sample_multiple``.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs of shape [num_inputs, ...].
        num_samples : int
            The number of samples to return in a set of samples.
        timeout : float
            A maximum time this primitive should take to sample outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        Sequence[Sequence[Output]]
            The multiple sets of samples of shape [num_inputs, num_samples, ...].
        """


class ProbabilisticCompositionalityMixin(Generic[Input, Output, Params]):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using probabilistic
    compositionality.

    This mixin adds methods to support at least:

    * Metropolis-Hastings

    Mixin should be used together with the ``PrimitiveBase`` class and ``SamplingCompositionalityMixin`` mixin.
    """

    @abc.abstractmethod
    def log_likelihood(self, *, output: Output, input: Input) -> float:
        """
        Returns log probability of output given input and params under this primitive:

        log(p(output | input, params))

        Parameters
        ----------
        output : Output
            The output.
        input : Input
            The input.

        Returns
        -------
        float
            log(p(output | input, params))
        """


class Scores(Generic[Params]):
    """
    A type representing a named tuple which holds all the differentiable fields from ``Params``.
    Their values are of type ``float``.
    """


class Gradients(Generic[Output]):
    """
    A type representing a structure of ``Output``, but the values are of type ``Optional[float]``.
    Value is ``None`` if gradient for that part of the structure is not possible.
    """


class GradientCompositionalityMixin(Generic[Input, Output, Params]):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using gradient-based
    compositionality.

    This mixin adds methods to support at least:

    * gradient-based, compositional end-to-end training
    * regularized pre-training
    * multi-task adaptation
    * black box variational inference
    * Hamiltonian Monte Carlo
    """

    @abc.abstractmethod
    def gradient_output(self, *, outputs: Sequence[Output], inputs: Sequence[Input]) -> Gradients[Output]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to output.

        When fit term temperature is set to non-zero, it should return the gradient with respect to output of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient
        of sum_i(log(p(output_i | input_i, params))) with respect to output.

        When fit term temperature is set to non-zero, it should return the gradient with respect to output of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs : Sequence[Output]
            The outputs.
        inputs : Sequence[Input]
            The inputs.

        Returns
        -------
        Gradients[Output]
            Gradients.
        """

    @abc.abstractmethod
    def gradient_params(self, *, outputs: Sequence[Output], inputs: Sequence[Input]) -> Scores[Params]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient of
        log(p(output | input, params)) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs : Sequence[Output]
            The outputs.
        inputs : Sequence[Input]
            The inputs.

        Returns
        -------
        Scores[Params]
            A named tuple with all fields from ``Params`` and values set to gradient for each parameter.
        """

    @abc.abstractmethod
    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        """
        Sets the temperature used in ``gradient_output`` and ``gradient_params``.

        Parameters
        ----------
        temperature : float
            The temperature to use, [0, inf), typically, [0, 1].
        """


class InspectLossMixin(metaclass=abc.ABCMeta):
    """
    Mixin which provides an abstract method for a caller to call to inspect which
    loss function a primitive is using internally.
    """

    @abc.abstractmethod
    def get_loss_function(self) -> Optional[str]:
        """
        Returns a D3M standard name of the loss function used by the primitive, or ``None`` if using
        a non-standard loss function or if the primitive does not use a loss function at all.

        Returns
        -------
        str
            A D3M standard name of the loss function used.
        """

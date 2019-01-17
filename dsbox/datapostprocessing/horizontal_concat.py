import pandas as pd
import time
import typing

import common_primitives.utils as common_utils
from d3m import container, exceptions
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.metadata.base import ALL_ELEMENTS
from copy import copy

from dsbox.datapreprocessing.cleaner import config

__all__ = ('HorizontalConcat',)


Inputs = container.DataFrame
Outputs = container.DataFrame


class HorizontalConcatHyperparams(hyperparams.Hyperparams):
    ignore_index = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether new df should use original index or not"
    )
    to_semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](""),
        default=("https://metadata.datadrivendiscovery.org/types/Attribute",
                 "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                 "https://metadata.datadrivendiscovery.org/types/CategoricalData"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Sementic typer to add for output dataframe"
    )
    column_name = hyperparams.Hyperparameter[int](
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=100,
        description="the control params for index name"
    )


class HorizontalConcat(TransformerPrimitiveBase[Inputs, Outputs, HorizontalConcatHyperparams]):
    """
        A primitive which concat a list of dataframe to a single dataframe horizontally,
        and it will also set metatdata for prediction, 
        we assume that inputs has same length
    """

    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-horizontal-concat",
        "version": config.VERSION,
        "name": "DSBox horizontal concat",
        "description": "horizontally concat a list of dataframe",
        "python_path": "d3m.primitives.dsbox.HorizontalConcat",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["concat", "horizontal"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: HorizontalConcatHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

    def produce(self, *, inputs1: Inputs, inputs2: Inputs,
                timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # need to rename inputs1.columns and inputs2.columns name
        if self.hyperparams["column_name"] == 0:
            left = inputs1.rename(columns={inputs1.columns[-1]: "0"})
        else:
            left = copy(inputs1)
        right = inputs2.rename(
            columns={inputs2.columns[-1]: str(self.hyperparams["column_name"]+1)})
        new_df = common_utils.horizontal_concat(left, right)

        for i, column in enumerate(new_df.columns):
            column_metadata = dict(new_df.metadata.query((ALL_ELEMENTS, i)))
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in column_metadata["semantic_types"]:
                column_metadata["semantic_types"] = self.hyperparams["to_semantic_types"]
                new_df.metadata = new_df.metadata.update(
                    (ALL_ELEMENTS, i), column_metadata)
        return CallResult(new_df)

# functions to fit in devel branch of d3m (2019-1-17)

    def set_training_data(self, *, inputs1: Inputs, inputs2: Inputs) -> None:
        pass

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs1: Inputs, inputs2: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        This method allows primitive author to implement an optimized version of both fitting
        and producing a primitive on same data.

        If any additional method arguments are added to primitive's ``set_training_data`` method
        or produce method(s), or removed from them, they have to be added to or removed from this
        method as well. This method should accept an union of all arguments accepted by primitive's
        ``set_training_data`` method and produce method(s) and then use them accordingly when
        computing results.

        The default implementation of this method just calls first ``set_training_data`` method,
        ``fit`` method, and all produce methods listed in ``produce_methods`` in order and is
        potentially inefficient.

        Parameters
        ----------
        produce_methods : Sequence[str]
            A list of names of produce methods to call.
        inputs : Inputs
            The inputs given to ``set_training_data`` and all produce methods.
        outputs : Outputs
            The outputs given to ``set_training_data``.
        timeout : float
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations : int
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        MultiCallResult
            A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs1, outputs=inputs2)

    def multi_produce(self, *, inputs1: Inputs, inputs2: Inputs, produce_methods: typing.Sequence[str],
                      timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        results = []
        for method_name in produce_methods:
            if method_name != 'produce' and not method_name.startswith('produce_'):
                raise exceptions.InvalidArgumentValueError(
                    "Invalid produce method name '{method_name}'.".format(method_name=method_name))

            if not hasattr(self, method_name):
                raise exceptions.InvalidArgumentValueError(
                    "Unknown produce method name '{method_name}'.".format(method_name=method_name))

            try:
                expected_arguments = set(self.metadata.query()['primitive_code'].get(
                    'instance_methods', {})[method_name]['arguments'])
            except KeyError as error:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown produce method name '{method_name}'.".format(method_name=method_name)) from error

            arguments = {'inputs1': inputs1,
                         'inputs2': inputs2,
                         }

            start = time.perf_counter()
            results.append(getattr(self, method_name)(
                timeout=timeout, **arguments))
            delta = time.perf_counter() - start

            # Decrease the amount of time available to other calls. This delegates responsibility
            # of raising a "TimeoutError" exception to produce methods themselves. It also assumes
            # that if one passes a negative timeout value to a produce method, it raises a
            # "TimeoutError" exception correctly.
            if timeout is not None:
                timeout -= delta

        # We return the maximum number of iterations done by any produce method we called.
        iterations_done = None
        for result in results:
            if result.iterations_done is not None:
                if iterations_done is None:
                    iterations_done = result.iterations_done
                else:
                    iterations_done = max(
                        iterations_done, result.iterations_done)

        return MultiCallResult(
            values={name: result.value for name,
                    result in zip(produce_methods, results)},
            has_finished=all(result.has_finished for result in results),
            iterations_done=iterations_done,
        )

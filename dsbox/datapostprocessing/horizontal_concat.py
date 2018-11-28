import pandas as pd
import time
import typing

import common_primitives.utils as common_utils
from d3m import container, exceptions
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.metadata.base import ALL_ELEMENTS


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

    def produce(self, *, inputs: Inputs, inputs1: Inputs, inputs2: Inputs,
                timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # need to rename inputs1.columns and inputs2.columns name
        inputslist = [inputs, inputs1, inputs2]
        from collections import deque
        combine_list = deque()
        for i,v in enumerate(inputslist):
            combine_list.append(v.rename(columns={v.columns[1]:str(str(v.columns[1])+"_"+str(i))}))
        while len(combine_list) != 1:
            left = combine_list.popleft()
            right = combine_list.popleft()
            combine_list.appendleft(common_utils.horizontal_concat(left, right))
        new_df = combine_list[0]

        for i, column in enumerate(new_df.columns):
            column_metadata = dict(new_df.metadata.query((ALL_ELEMENTS, i)))
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in column_metadata["semantic_types"]:
                column_metadata["semantic_types"] = self.hyperparams["to_semantic_types"]
                new_df.metadata = new_df.metadata.update(
                    (ALL_ELEMENTS, i), column_metadata)
        return CallResult(new_df)

    def multi_produce(self, *, inputs: Inputs, inputs1: Inputs, inputs2: Inputs, produce_methods: typing.Sequence[str],
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

            arguments = {'inputs': inputs,
                         'inputs1': inputs1,
                         'inputs2': inputs2}

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

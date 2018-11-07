import pandas as pd
import time
import typing

import common_primitives.utils as common_utils
from d3m import container, exceptions
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult, MultiCallResult

from dsbox.datapreprocessing.cleaner import config

__all__ = ('VerticalConcat',)

# Inputs = container.List
Inputs = container.DataFrame
Outputs = container.DataFrame


class VerticalConcatHyperparams(hyperparams.Hyperparams):
    ignore_index = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether new df should use original index or not"
    )
    sort_on_primary_key = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether new df will be sorted based on primary key, mostly time it will be d3mIndex"
    )


class VerticalConcat(TransformerPrimitiveBase[Inputs, Outputs, VerticalConcatHyperparams]):
    """
        A primitive which concat a list of dataframe to a single dataframe vertically
    """

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-vertical-concat",
        "version": config.VERSION,
        "name": "DSBox vertically concat",
        "description": "A primitive which concat a list of dataframe to a single dataframe vertically",
        "python_path": "d3m.primitives.dsbox.VerticalConcat",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["concat", "vertical"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: VerticalConcatHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._training_data = None
        self._fitted = False

    def produce(self, *, inputs: Inputs, inputs1: Inputs, inputs2: Inputs,
                timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        new_df = pd.concat([x for x in [inputs, inputs1, inputs2] if x is not None], ignore_index=self.hyperparams["ignore_index"])
        if self.hyperparams["sort_on_primary_key"]:
            primary_key_col = common_utils.list_columns_with_semantic_types(metadata=new_df.metadata, semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/PrimaryKey"])
            if not primary_key_col:
                warnings.warn("No PrimaryKey column found. Will not sort on PrimaryKey")
                return CallResult(self._update_metadata(new_df))
            new_df = new_df.sort_values([new_df.columns[pos] for pos in primary_key_col])
        return CallResult(self._update_metadata(new_df))

    def multi_produce(self, *, inputs: Inputs, inputs1: Inputs, inputs2: Inputs, produce_methods: typing.Sequence[str],
                timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        results = []
        for method_name in produce_methods:
            if method_name != 'produce' and not method_name.startswith('produce_'):
                raise exceptions.InvalidArgumentValueError("Invalid produce method name '{method_name}'.".format(method_name=method_name))

            if not hasattr(self, method_name):
                raise exceptions.InvalidArgumentValueError("Unknown produce method name '{method_name}'.".format(method_name=method_name))

            try:
                expected_arguments = set(self.metadata.query()['primitive_code'].get('instance_methods', {})[method_name]['arguments'])
            except KeyError as error:
                raise exceptions.InvalidArgumentValueError("Unknown produce method name '{method_name}'.".format(method_name=method_name)) from error

            arguments = {'inputs': inputs,
                         'inputs1': inputs1,
                         'inputs2': inputs2}

            start = time.perf_counter()
            results.append(getattr(self, method_name)(timeout=timeout, iterations=iterations, **arguments))
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
                    iterations_done = max(iterations_done, result.iterations_done)

        return MultiCallResult(
            values={name: result.value for name, result in zip(produce_methods, results)},
            has_finished=all(result.has_finished for result in results),
            iterations_done=iterations_done,
        )


    @staticmethod
    def _update_metadata(df: container.DataFrame) -> container.DataFrame:
        old_metadata = dict(df.metadata.query(()))
        old_metadata["dimension"] = dict(old_metadata["dimension"])
        old_metadata["dimension"]["length"] = df.shape[0]
        df.metadata = df.metadata.update((), old_metadata)
        return df

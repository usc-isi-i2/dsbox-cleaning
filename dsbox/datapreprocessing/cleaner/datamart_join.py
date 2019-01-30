import typing

# importing d3m stuff
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config
import time

# import datamart stuff
# from datamart.augment import Augment 

Inputs1 = DataFrame
Inputs2 = DataFrame
Outputs = DataFrame


# join two dataframe by columns

class DatamartJoinHyperparams(hyperparams.Hyperparams):
    left_columns = hyperparams.Hyperparameter[typing.List[typing.List[int or str]]](
        default=[[1], []],  # fixme
        description='columns to join for the left_colums',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    right_columns = hyperparams.Hyperparameter[typing.List[typing.List[int or str]]](
        default=[[1], []],  # fixme
        description='columns to join for the right_colums',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class DatamartJoin(TransformerPrimitiveBase[Inputs1, Inputs2, DatamartJoinHyperparams]):
    '''
    A primitive perform join between datasets by lists of column names 
    '''

    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "datamart-augmentation",
        "version": config.VERSION,
        "name": "Datamart Augmentation",
        "python_path": "d3m.primitives.data_augmentation.Join.DSBOX",
        "primitive_family": "DATA_AUGMENTATION",
        "algorithm_types": ["APPROXIMATE_DATA_AUGMENTATION"],  # fix me!
        "keywords": ["data augmentation", "datamart", "join"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        # 'precondition': [],
        # 'hyperparams_to_tune': []

    })

    def __init__(self, *, hyperparams: DatamartJoinHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iteration_done = False

    def produce(self, *, inputs1: Inputs1, inputs2: Inputs2, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        res_df = DataFrame(join(left_df=inputs1, right_df=inputs2, left_columns=self.hyperparams["left_columns"], right_columns=sefl.hyperparams["right_columns"]))
        self._has_finished = True
        self._iteration_done = True
        return CallResult(res_df)


    # functions to fit in devel branch of d3m (2019-1-17)

    def set_training_data(self, *, inputs1: Inputs1, inputs2: Inputs2) -> None:
        pass

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs1: Inputs1, inputs2: Inputs2, timeout: float = None, iterations: int = None) -> MultiCallResult:
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

    def multi_produce(self, *, inputs1: Inputs1, inputs2: Inputs2, produce_methods: typing.Sequence[str],
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

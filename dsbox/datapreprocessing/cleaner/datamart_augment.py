import typing

# importing d3m stuff
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config

# field for importing datamart stuff
from datamart import augment
from datamart import Dataset

Inputs1 = List
Inputs2 = DataFrame  # FIXMEE
Outputs = DataFrame


class DatamartAugmentationHyperparams(hyperparams.Hyperparams):
    # indexes of dataset to choose from
    #
    n_index = hyperparams.Hyperparameter[int](
        default=0,
        description='index of dataset from list to choose from. Default is 0',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class DatamartAugmentation(TransformerPrimitiveBase[Inputs1, Inputs2, DatamartAugmentationHyperparams]):
    '''
    A primitive that takes a list of datamart dataset and choose 1 or a few best dataframe and perform join, return an accessible d3m.dataframe for further processing
    '''
    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "datamart-augmentation",
        "version": config.VERSION,
        "name": "Datamart Augmentation",
        "python_path": "d3m.primitives.datamart.Augmentation",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": ["AUDIO_STREAM_MANIPULATION"], # fix me!
        "keywords": ["data augmentation", "datamart", "join"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        'precondition': [],
        'hyperparms_to_tune': []

    })

    def __init__(self, *, hyperparams: DatamartAugmentationHyperparams)-> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False

    def produce(self, *, inputs1: Inputs1, inputs2: Inputs2, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # sort the inputslist by best score
        inputs1.sort(key=lambda x: x.score, reverse=True)
        # choose the best one? maybe more, determined by hyperparms
        res_df = augment(
            original_data=inputs2, augment_data=inputs1[self.hyperparms["n_index"]])  # a pd.dataframe

        # join with inputs2

        # updating "attribute columns", "datatype" from datamart.Dataset

        self._has_finished = True
        self._iterations_done = True
        return CallResult(res_df)

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

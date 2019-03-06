import typing
import importlib
import logging

# importing d3m stuff
from d3m import exceptions
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config
import time

# # field for importing datamart stuff
# from datamart import augment
# from datamart import Dataset

# # fixme, now datamart_nyu and datamart_isi has the same import module name "datamart"
# from datamart_nyu import augment

Inputs1 = List
Inputs2 = DataFrame  # FIXME
Outputs = DataFrame
_logger = logging.getLogger(__name__)

class DatamartAugmentationHyperparams(hyperparams.Hyperparams):
    # indexes of dataset to choose from
    #

    url = hyperparams.Hyperparameter[str](
        default='https://isi-datamart.edu',
        description='url indicates which datamart resource to use',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

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
        "python_path": "d3m.primitives.data_augmentation.Augmentation.DSBOX",
        "description": "Join the given dataframe with the highest ranked datamart dataset return by the primitive QueryFromDataframe. Also, see the DatamartJoin primitive for joining two dataframes.",
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

    def __init__(self, *, hyperparams: DatamartAugmentationHyperparams)-> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False

    def _import_module(self):
        if self.hyperparams["url"].startswith('https://isi-datamart.edu'):
            global ISI_datamart
            ISI_datamart = importlib.import_module('datamart')
            # ISI_Dataset = importlib.import_module('datamart.Dataset')
            return 1
        if self.hyperparams["url"].startswith('https://datamart.d3m.vida-nyu.org'):
            # from datamart_nyu import augment, Dataset
            global NYU_datamart
            NYU_datamart = importlib.import_module('datamart_nyu')
            # NYU_Dataset = importlib.import_module('datamart_nyu.Dataset')
            return 2
        return 0

    def produce(self, *, inputs1: Inputs1, inputs2: Inputs2, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        status = self._import_module()
        if status == 0:
            _logger.error("not a valid  url")
            return CallResult(DataFrame())
        if status == 1:  # run isi-datamart
            # sort the inputslist by best score
            inputs1.sort(key=lambda x: x.score, reverse=True)
            # choose the best one? maybe more, determined by hyperparams
            res_df = ISI_datamart.augment(
                original_data=inputs2, augment_data=inputs1[self.hyperparams["n_index"]])  # a pd.dataframe

            # join with inputs2

            # updating "attribute columns", "datatype" from datamart.Dataset
        else:  # run
            inputs1.sort(key=lambda x: x.score, reverse=True)
            res_df = NYU_datamart.augment(
                data=inputs2, augment_data=inputs1[self.hyperparams["n_index"]])

        self._has_finished = True
        self._iterations_done = True
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

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs1=inputs1, inputs2=inputs2) # add check for inputs name here, must be the sames as produce and set_training data

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

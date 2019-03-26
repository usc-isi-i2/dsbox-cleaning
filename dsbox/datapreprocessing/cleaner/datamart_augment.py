import typing
import importlib
import logging
import frozendict
import copy
# importing d3m stuff
from d3m import exceptions
from d3m import container
from d3m.metadata.base import Metadata, DataMetadata, ALL_ELEMENTS
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from common_primitives import utils
from . import config
import time

# # field for importing datamart stuff
# from datamart import augment
# from datamart import Dataset

# # fixme, now datamart_nyu and datamart_isi has the same import module name "datamart"
# from datamart_nyu import augment

Inputs1 = container.List
Inputs2 = container.Dataset  # FIXME
Outputs = container.Dataset
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

    join_type = hyperparams.Hyperparameter[str](
        default="exact",
        description="joiner to use(exact, or approximate)",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']

    )
    joining_columns = hyperparams.Hyperparameter[str](
        default='',
        description='column names used for setting up the blocking for join',
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
        "python_path": "d3m.primitives.data_augmentation.datamart_augmentation.DSBOX",
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
        input_dataset = inputs2
        self._res_id, _ = utils.get_tabular_resource(input_dataset, None, has_hyperparameter=False)
        input_main_df = input_dataset[self._res_id]
        if status == 0:
            _logger.error("not a valid  url")
            return CallResult(None, True, 1)
        if status == 1:  # run isi-datamart
            # sort the inputslist by best score
            join_type = self.hyperparams["join_type"]
            if join_type == "exact":
                joiner = ISI_datamart.joiners.joiner_base.JoinerType.EXACT_MATCH
            elif join_type == "rltk":
                joiner = ISI_datamart.joiners.joiner_base.JoinerType.RLTK
            inputs1.sort(key=lambda x: x.score, reverse=True)
            # choose the best one? maybe more, determined by hyperparams
            res_df = ISI_datamart.augment(
                original_data=input_main_df,
                augment_data=inputs1[self.hyperparams["n_index"]],
                joining_columns = self.hyperparams["joining_columns"],
                joiner=joiner)  # a pd.dataframe

            # join with inputs2
            import pdb
            pdb.set_trace()
            result = self._generate_new_metadata(df_joined = res_df.df, input_dataset = input_dataset, augment_dataset = inputs1)
            # updating "attribute columns", "datatype" from datamart.Dataset
        else:  # run
            inputs1.sort(key=lambda x: x.score, reverse=True)
            res_df = NYU_datamart.augment(
                data=inputs2, augment_data=inputs1[self.hyperparams["n_index"]])

        self._has_finished = True
        self._iterations_done = True
        return CallResult(result, True, 1)

    def _generate_new_metadata(self, df_joined, input_dataset, augment_dataset): 
# adjust column position, to put d3mIndex at first columns and put target at last columns
        columns_all = list(df_joined.columns)
        if 'd3mIndex' in df_joined.columns:
            oldindex = columns_all.index('d3mIndex')
            columns_all.insert(0, columns_all.pop(oldindex))
        else:
            _logger.error("No d3mIndex column found after data-mart augment!!!")
        target_amount = 0
        metadata_new_target = {}
        new_target_column_number = []
        # targets_from_problem = input_problem_meta.query(())["inputs"]["data"][0]["targets"]
        # for t in targets_from_problem:
        #     oldindex = columns_all.index(t["colName"])
        #     target_amount += 1
        #     temp = columns_all.pop(oldindex)
        #     new_target_column_number.append(len(columns_all))
        #     metadata_new_target[t["colName"]] = len(columns_all)
        #     columns_all.insert(len(columns_all), temp)
        df_joined = df_joined[columns_all]

        # start adding metadata for dataset
        metadata_dict_left = {}
        metadata_dict_right = {}
        for each in augment_dataset[0].metadata['variables']:
            decript = each['description']
            dtype = decript.split("dtype: ")[-1]
            if "float" in dtype:
                semantic_types = (
                      "http://schema.org/Float",
                      "https://metadata.datadrivendiscovery.org/types/Attribute"
                     )
            elif "int" in dtype:
                semantic_types = (
                      "http://schema.org/Integer",
                      "https://metadata.datadrivendiscovery.org/types/Attribute"
                     )
            else:
                semantic_types = (
                      "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                      "https://metadata.datadrivendiscovery.org/types/Attribute"
                     )
            each_meta = {
                "name": each['name'],
                "structural_type": str,
                "semantic_types": semantic_types,
                "description": decript
            }
            metadata_dict_right[each['name']] = frozendict.FrozenOrderedDict(each_meta)

        left_df_column_legth = input_dataset.metadata.query((self._res_id, ALL_ELEMENTS))['dimension']['length']
        for i in range(left_df_column_legth):
            each_selector = (self._res_id, ALL_ELEMENTS, i)
            each_column_meta = input_dataset.metadata.query(each_selector)
            metadata_dict_left[each_column_meta['name']] = each_column_meta

        metadata_new = DataMetadata()
        metadata_old = copy.copy(input_dataset.metadata)
        new_column_meta = dict(input_dataset.metadata.query((self._res_id, ALL_ELEMENTS)))
        new_column_meta['dimension'] = dict(new_column_meta['dimension'])
        new_column_meta['dimension']['length'] = df_joined.shape[1]
        new_row_meta = dict(input_dataset.metadata.query((self._res_id,)))
        new_row_meta['dimension'] = dict(new_row_meta['dimension'])
        new_row_meta['dimension']['length'] = df_joined.shape[0]
        # update whole source description
        metadata_new = metadata_new.update((), metadata_old.query(()))
        metadata_new = metadata_new.update((self._res_id,), new_row_meta)
        metadata_new = metadata_new.update((self._res_id, ALL_ELEMENTS), new_column_meta)

        new_column_names_list = list(df_joined.columns)
        # update each column's metadata
        for i, current_column_name in enumerate(new_column_names_list):
            each_selector = (self._res_id, ALL_ELEMENTS, i)
            if current_column_name in metadata_dict_left:
                new_metadata_i = metadata_dict_left[current_column_name]
            else:
                new_metadata_i = metadata_dict_right[current_column_name]
            metadata_new = metadata_new.update(each_selector, new_metadata_i)
        # end adding metadata for dataset
        augmented_dataset = input_dataset.copy()
        augmented_dataset.metadata = metadata_new
        df_joined_d3m = container.DataFrame(df_joined, generate_metadata=False, dtype = str)
        augmented_dataset[self._res_id] = df_joined_d3m
        # end updating dataset

        return augmented_dataset




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

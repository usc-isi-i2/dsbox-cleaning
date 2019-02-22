import copy
import collections
import typing
import logging
import enum
import sys
import random
import itertools

from d3m.container import Dataset
from common_primitives.dataset_remove_columns import RemoveColumnsPrimitive # _select_columns_metadata
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params, base as metadata_base
from common_primitives import utils
from . import config

Input = Dataset
Output = Dataset

class Status(enum.Enum):
    UNFIT = 0
    TRAIN = 1
    TEST = 2

class Params(params.Params):
    status: int
    need_reduce_rows: bool
    need_reduce_columns: bool
    columns_remained: typing.List[object]
    rows_remained: typing.List[object]

class SplitterHyperparameter(hyperparams.Hyperparams):
    threshold_columns_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=300,
        description='The threshold value of amount of columns in a dataframe, if the value is larger, it will be splitted (sampled).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    threshold_rows_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=100000,
        description='The threshold value of amount of rows in a dataframe, if the value is larger, it will be splitted (sampled).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    further_reduce_threshold_columnes_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=20,
        description='The threshold of columns amount to further reduce the threshold_rows_length value for the condition that both the amount of columns and rows are very large',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    further_reduce_ratio = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.3,
        upper_inclusive = True,
        description='The ratio to further reduce the threshold_rows_length value for the condition that both the amount of columns and rows are very large',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    random_seed_val = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=4676,
        description='The random seed for generating the sampling results, set up for consistent results.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Splitter(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, SplitterHyperparameter]):
    """
    A primitive that could be used before processing the dataset.
    If the size of the dataset(or dataframe) is smaller than threshold, it will do nothing but pass through the original dataset
    If the size if larger than the threshold, it will reduce the amount of columns or rows or both by splitting the dataset/dataframe.
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "DSBox-splitter",
        "version": config.VERSION,
        "name": "DSBox Splitter",
        "description": "Reduce the dataset amount if necessary",
        "python_path": "d3m.primitives.data_preprocessing.Splitter.DSBOX",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [ config.REPOSITORY ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "preprocessing",  "split"],
        "installation": [ config.INSTALLATION ],
        #"location_uris": [],
        #"precondition": [],
        #"effects": [],
        #"hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: SplitterHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)

        self._logger = logging.getLogger(__name__)
        self.hyperparams = hyperparams
        # set up random seed for consistence
        random.seed(hyperparams['random_seed_val'])
        self._threshold_columns_length = self.hyperparams['threshold_columns_length']
        self._threshold_rows_length = self.hyperparams['threshold_rows_length']
        self._further_reduce_threshold_columnes_length = self.hyperparams['further_reduce_threshold_columnes_length']
        self._further_reduce_ratio = self.hyperparams['further_reduce_ratio']

        self._status = Status.UNFIT
        self._main_resource_id = None
        self._need_reduce_columns = False
        self._need_reduce_rows = False
        self._training_inputs = None
        self._fitted = False


    def get_params(self) -> Params:
        param = Params(
                       status = self._status,
                       need_reduce_rows = self._need_reduce_rows,
                       need_reduce_columns = self._need_reduce_columns,
                       columns_remained = self._columns_remained,
                       rows_remained = self._rows_remained,
                       index_removed_percent = self._index_removed_percent,
                       main_resource_id = self._main_resource_id
                      )
        return param

    def set_params(self, *, params: Params) -> None:
        self._status = params['status']
        self._need_reduce_columns = params['need_reduce_columns']
        self._need_reduce_rows = params['need_reduce_rows']
        self._columns_remained = params['columns_remained']
        self._rows_remained = params['rows_remained']
        self._index_removed_percent = params['index_removed_percent']
        self._main_resource_id = params['main_resource_id']
        self._fitted = True

    def set_training_data(self, *, inputs: Input) -> None:
        self._training_inputs = inputs
        main_resource_id, main_resource = utils.get_tabular_resource(inputs, None, has_hyperparameter=False)
        self._main_resource_id = main_resource_id
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        check the shape of the main resource dataset. I
        f the size is larger than threshold, the primitive will record and generate 
        a list of columns/ rows that need to be remained.
        """
        if self._fitted or self._status is not Status.UNFIT:
            return

        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        data = self._training_inputs.copy()
        main_res_shape = data[self._main_resource_id].shape

        if main_res_shape[0] > self._threshold_rows_length:
            self._need_reduce_rows = True
            if main_res_shape[1] > self._further_reduce_threshold_columnes_length:
                self._threshold_rows_length = self._threshold_rows_length * self._further_reduce_ratio
            self._logger.info("This dataset's columns number and rows number are both oversized, will further reduce the threshold of the rows about to be." + str(self._threshold_rows_length))

        if main_res_shape[1] > self._threshold_columns_length:
            self._need_reduce_columns = True

        if self._need_reduce_columns and self._need_reduce_rows:
            self._logger.info("This dataset's columns number and rows number are both oversized, will sample both of them.")
        elif self._need_reduce_columns:
            self._logger.info("The columns number of the input dataset is very large, will split part of them.")
        elif self._need_reduce_rows:
            self._logger.info("The rows number of the input dataset is very large, will split part of them.")
        else:
            self._logger.info("This dataset's size is OK, no split on dataset needed.")

        # copy from d3m here, what is this used for?
        # Graph is the adjacency representation for the relations graph. Make it not be a "defaultdict".
        # self._graph = dict(utils.build_relation_graph(self._training_inputs))

        self._status = Status.TRAIN
        self._fitted = True
        return CallResult(None, has_finished=True, iterations_done=1)


    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        sample the dataset if needed
        """
        if not self._fitted or self._status is Status.UNFIT:
            raise ValueError('Splitter model not fitted. Use fit() first')

        # Return if there is nothing to split
        if not self._need_reduce_rows and not self._need_reduce_columns:
            return CallResult(inputs, True, 1)

        results = copy.copy(inputs)
        if self._status is Status.TEST:
            self._logger.info("In test process, no split on rows needed") 
            return CallResult(results, True, 1)

        else:
            if self._need_reduce_rows:
                results = self._split_rows(results)
            if self._need_reduce_columns:
                results = self._split_columns(results)
            # if it is first time to run produce here, we should in train status
            # so we need to let the program know that for next time, we will in test process
            if self._status is Status.TRAIN:
                self._status = Status.TEST

        return CallResult(results, True, 1)

    def _split_rows(self, input_dataset):
        """
            Inner function to sample part of the rows of the input dataset
            adapted from d3m's common-primitives

        Returns
        -------
        Dataset
            The sampled Dataset
        """
        rows_length = input_dataset[self._main_resource_id].shape[0]
        all_indexes_list = range(1, rows_length)
        sample_indices = random.sample(all_indexes_list, self._threshold_columns_length)
        # We store rows as sets, but later on we sort them when we cut DataFrames.
        row_indices_to_keep_sets: typing.Dict[str, typing.Set[int]] = collections.defaultdict(set)
        row_indices_to_keep_sets[self._main_resource_id] = set(sample_indices)

        # We sort indices to get deterministic outputs from sets (which do not have deterministic order).
        row_indices_to_keep = {resource_id: sorted(indices) for resource_id, indices in row_indices_to_keep_sets.items()}

        output_dataset = utils.cut_dataset(input_dataset, row_indices_to_keep)

        return output_dataset

    def _split_columns(self, inputs):
        """
            Inner function to sample part of the columns of the input dataset
        """
        input_dataset_shape = inputs[self._main_resource_id].shape
        # find target columns, we should not split these columns
        target_columns = utils.list_columns_with_semantic_types(self._training_inputs.metadata, ['https://metadata.datadrivendiscovery.org/types/TrueTarget'], at=(self._main_resource_id,))
        if not target_columns:
            self._logger.warn("No target columns found from the input dataset.")
        index_columns = utils.get_index_columns(self._training_inputs.metadata,at=(self._main_resource_id,))
        if not index_columns:
            self._logger.warn("No index columns found from the input dataset.")

        # check again on the amount of the attributes columns only
        # we only need to sample when attribute column numbers are larger than threshould
        attribute_column_length = (input_dataset_shape[1] - len(index_columns) - len(target_columns))
        if attribute_column_length > self._threshold_columns_length:
            attribute_columns = set(range(input_dataset_shape[1]))
            for each_target_column in target_columns:
                attribute_columns.remove(each_target_column)
            for each_index_column in index_columns:
                attribute_columns.remove(each_index_column)

            # generate the remained column index randomly and sort it
            remained_columns = random.sample(attribute_columns, self._threshold_columns_length)
            remained_columns.extend(target_columns)
            remained_columns.extend(index_columns)
            remained_columns.sort()
            # use common primitive's RemoveColumnsPrimitive inner function to finish sampling
            outputs = copy.copy(inputs)
            # Just to make sure.
            outputs.metadata = inputs.metadata.set_for_value(outputs, generate_metadata=False)
            outputs[self._main_resource_id] = inputs[self._main_resource_id].iloc[:, remained_columns]
            outputs.metadata = RemoveColumnsPrimitive._select_columns_metadata(outputs.metadata, self._main_resource_id, remained_columns)

        return outputs
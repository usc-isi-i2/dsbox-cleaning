import copy
import collections
import typing
import logging
import enum
import sys
import random
import itertools

from d3m.container import Dataset
from common_primitives.dataset_remove_columns import RemoveColumnsPrimitive # _select_column_metadata
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
    status: Status
    need_reduce_row: bool
    need_reduce_column: bool
    main_resource_id: str
    column_remained: typing.List[object]
    row_remained: typing.Dict

class SplitterHyperparameter(hyperparams.Hyperparams):
    threshold_column_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=300,
        description='The threshold value of amount of column in a dataframe, if the value is larger, it will be splitted (sampled).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    threshold_row_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=100000,
        description='The threshold value of amount of row in a dataframe, if the value is larger, it will be splitted (sampled).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    further_reduce_threshold_column_length = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=200,
        description='The threshold of column amount to further reduce the threshold_row_length value for the condition that both the amount of column and row are very large',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    further_reduce_ratio = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.5,
        upper_inclusive = True,
        description='The ratio to further reduce the threshold_row_length value for the condition that both the amount of column and row are very large',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    # random_seed_val = hyperparams.UniformInt(
    #     lower=0,
    #     upper=sys.maxsize,
    #     default=4676,
    #     description='The random seed for generating the sampling results, set up for consistent results.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    # )

class Splitter(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, SplitterHyperparameter]):
    """
    A primitive that could be used before processing the dataset.
    If the size of the dataset(or dataframe) is smaller than threshold, it will do nothing but pass through the original dataset
    If the size if larger than the threshold, it will reduce the amount of column or row or both by splitting the dataset/dataframe.
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

    def __init__(self, *, hyperparams: SplitterHyperparameter, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._logger = logging.getLogger(__name__)
        self.hyperparams = hyperparams
        # set up random seed for consistence
        # random.seed(hyperparams['random_seed_val'])
        random.seed(random_seed)
        self._threshold_column_length = self.hyperparams['threshold_column_length']
        self._threshold_row_length = self.hyperparams['threshold_row_length']
        self._further_reduce_threshold_column_length = self.hyperparams['further_reduce_threshold_column_length']
        self._further_reduce_ratio = self.hyperparams['further_reduce_ratio']

        self._column_remained = []
        self._row_remained = {}
        self._status = Status.UNFIT
        self._main_resource_id = ""
        self._need_reduce_column = False
        self._need_reduce_row = False

        self._training_inputs = None
        self._fitted = False


    def get_params(self) -> Params:
        param = Params(
                       status = self._status,
                       need_reduce_row = self._need_reduce_row,
                       need_reduce_column = self._need_reduce_column,
                       column_remained = self._column_remained,
                       row_remained = self._row_remained,
                       main_resource_id = self._main_resource_id
                      )
        return param

    def set_params(self, *, params: Params) -> None:
        self._status = params['status']
        self._need_reduce_column = params['need_reduce_column']
        self._need_reduce_row = params['need_reduce_row']
        self._column_remained = params['column_remained']
        self._row_remained = params['row_remained']
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
        a list of column/ row that need to be remained.
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        data = self._training_inputs.copy()
        main_res_shape = data[self._main_resource_id].shape

        if main_res_shape[0] > self._threshold_row_length:
            self._need_reduce_row = True
            if main_res_shape[1] > self._further_reduce_threshold_column_length:
                self._threshold_row_length = self._threshold_row_length * self._further_reduce_ratio
                self._logger.info("This dataset's column number and row number are both oversized, will further reduce the threshold of the row about to be." + str(self._threshold_row_length))

        if main_res_shape[1] > self._threshold_column_length:
            self._need_reduce_column = True

        if self._need_reduce_column and self._need_reduce_row:
            self._logger.info("This dataset's column number and row number are both oversized, will sample both of them.")
        elif self._need_reduce_column:
            self._logger.info("The column number of the input dataset is very large, will split part of them.")
        elif self._need_reduce_row:
            self._logger.info("The row number of the input dataset is very large, will split part of them.")
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
        if not self._need_reduce_row and not self._need_reduce_column:
            self._logger.info("No sampling need.")
            return CallResult(inputs, True, 1)
        else:
            self._logger.info("sampling needed.")

        results = copy.copy(inputs)
        if self._status is Status.TEST:
            self._logger.info("In test process, no split on row needed") 
            return CallResult(results, True, 1)

        else:
            if self._need_reduce_row:
                self._logger.info("Now sampling rows.") 
                results = self._split_row(results)
                self._logger.info("Sampling rows finished.") 
            if self._need_reduce_column:
                self._logger.info("Now sampling columns.") 
                results = self._split_column(results)
                self._logger.info("Sampling columns finished.") 
            # if it is first time to run produce here, we should in train status
            # so we need to let the program know that for next time, we will in test process
            if self._status is Status.TRAIN:
                self._status = Status.TEST

        self._logger.info("After sampling, the dataset's main resource shape is:",results[self._main_resource_id].shape)
        return CallResult(results, True, 1)

    def _split_row(self, input_dataset):
        """
            Inner function to sample part of the row of the input dataset
            adapted from d3m's common-primitives

        Returns
        -------
        Dataset
            The sampled Dataset
        """
        row_length = input_dataset[self._main_resource_id].shape[0]
        all_indexes_list = range(1, row_length)
        sample_indices = random.sample(all_indexes_list, self._threshold_row_length)
        # We store row as sets, but later on we sort them when we cut DataFrames.
        row_indices_to_keep_sets: typing.Dict[str, typing.Set[int]] = collections.defaultdict(set)
        row_indices_to_keep_sets[self._main_resource_id] = set(sample_indices)

        # We sort indices to get deterministic outputs from sets (which do not have deterministic order).
        self._row_remained = {resource_id: sorted(indices) for resource_id, indices in row_indices_to_keep_sets.items()}
        output_dataset = utils.cut_dataset(input_dataset, self._row_remained)

        return output_dataset

    def _split_column(self, inputs):
        """
            Inner function to sample part of the column of the input dataset
        """
        input_dataset_shape = inputs[self._main_resource_id].shape
        # find target column, we should not split these column
        target_column = utils.list_column_with_semantic_types(self._training_inputs.metadata, ['https://metadata.datadrivendiscovery.org/types/TrueTarget'], at=(self._main_resource_id,))
        if not target_column:
            self._logger.warn("No target column found from the input dataset.")
        index_column = utils.get_index_column(self._training_inputs.metadata,at=(self._main_resource_id,))
        if not index_column:
            self._logger.warn("No index column found from the input dataset.")

        outputs = copy.copy(inputs)
        if self._status is Status.TRAIN:
            # check again on the amount of the attributes column only
            # we only need to sample when attribute column numbers are larger than threshould
            attribute_column_length = (input_dataset_shape[1] - len(index_column) - len(target_column))
            if attribute_column_length > self._threshold_column_length:
                attribute_column = set(range(input_dataset_shape[1]))
                for each_target_column in target_column:
                    attribute_column.remove(each_target_column)
                for each_index_column in index_column:
                    attribute_column.remove(each_index_column)

                # generate the remained column index randomly and sort it
                self._column_remained = random.sample(attribute_column, self._threshold_column_length)
                self._column_remained.extend(target_column)
                self._column_remained.extend(index_column)
                self._column_remained.sort()
            # use common primitive's RemoveColumnsPrimitive inner function to finish sampling

        if len(self._column_remained) > 0: 
            # Just to make sure.
            outputs.metadata = inputs.metadata.set_for_value(outputs, generate_metadata=False)
            outputs[self._main_resource_id] = inputs[self._main_resource_id].iloc[:, self._column_remained]
            outputs.metadata = RemoveColumnsPrimitive._select_column_metadata(outputs.metadata, self._main_resource_id, self._column_remained)

        return outputs
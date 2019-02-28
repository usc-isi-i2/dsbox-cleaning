import typing

# importing d3m stuff
from d3m import exceptions
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config
import time

# import datamart stuff
from datamart import join
from datamart.joiners.joiner_base import JoinerType
import pandas as pd
# from datamart.augment import Augment

Inputs = List
Outputs = DataFrame


# join two dataframe by columns

class DatamartJoinHyperparams(hyperparams.Hyperparams):
    left_columns = hyperparams.Hyperparameter[typing.List[typing.List[int]]](
        default=[[1]],  # fixme
        description='columns to join for the left_colums',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    right_columns = hyperparams.Hyperparameter[typing.List[typing.List[int]]](
        default=[[1]],  # fixme
        description='columns to join for the right_colums',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    join_type = hyperparams.Hyperparameter[str](
        default="exact",
        description="joiner to use(exact, or approximate)",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']

    )


class DatamartJoin(TransformerPrimitiveBase[Inputs, Outputs, DatamartJoinHyperparams]):
    '''
    A primitive perform join between datasets by lists of column names
    '''

    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "datamart-join",
        "version": config.VERSION,
        "name": "Datamart Augmentation",
        "python_path": "d3m.primitives.data_augmentation.Join.DSBOX",
        "description": "Joins two dataframes into one dataframe. The primtive takes two dataframes, left_dataframe and right_dataframe, and two lists specifing the join columns, left_columns and right_columns.",
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

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        left_df, right_df = inputs
        left_df = pd.DataFrame(left_df)
        right_df = pd.DataFrame(right_df) #!
        join_type = self.hyperparams["join_type"]
        if join_type == "exact":
            joiner = JoinerType.EXACT_MATCH
        else:
            joiner = JoinerType.RLTK

        res = join(left_data=left_df, right_data=right_df,
                                left_columns=self.hyperparams["left_columns"], right_columns=self.hyperparams["right_columns"], joiner=joiner)
        res_df = DataFrame(res.df)
        self._has_finished = True
        self._iteration_done = True
        return CallResult(res_df, True, 1)

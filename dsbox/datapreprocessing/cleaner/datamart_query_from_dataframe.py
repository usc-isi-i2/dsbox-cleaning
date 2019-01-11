# input: dataframe, can be

import pandas as pd
import typing


# importing d3m stuff
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config

# field for importing datamart stuff
from datamart import search
from datamart import Dataset


Inputs = DataFrame
Outputs = List


class QueryFromDataFrameHyperparams(hyperparams.Hyperparams):

    # query
    query = hyperparams.Hyperparameter[dict](
        default={},
        description="The query to execute, default blank",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    # may add new hyperparams


class QueryFromDataframe(TransformerPrimitiveBase[Inputs, Outputs, QueryFromDataFrameHyperparams]):
    '''
    A primitive that takes a DataFrame and output a list of Autmented datamart dataset
    '''
    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "datamart-from-dataframe",
        "version": config.VERSION,
        "name": "Datamart Query Primitive from dataframe",
        "python_path": "d3m.primitives.datamart.QueryDataframe",  # FIXME
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": ["AUDIO_STREAM_MANIPULATION"],  # FIXME!
        "keywords": ["data augmentation", "datamart"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        'precondition': [],
        'hyperparms_to_tune': []

    })

    def __init__(self, *, hyperparams: QueryFromDataFrameHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # fixme one of the field
        res_list = search(
            query=self.hyperparams["query"], data=inputs)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(res_list)

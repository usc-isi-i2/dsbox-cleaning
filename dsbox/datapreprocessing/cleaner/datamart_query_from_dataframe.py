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



Inputs = DataFrame
Outputs = List


class QueryFromDataFrameHyperparams(hyperparams.Hyperparams):

    url = hyperparams.Hyperparameter[str](
        default='https://isi-datamart.edu',
        description='url indicates which datamart resource to use',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
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
        "python_path": "d3m.primitives.dsbox.QueryDataframe",
        "primitive_family": "DATA_AUGMENTATION",
        "algorithm_types": ["APPROXIMATE_DATA_AUGMENTATION"],
        "keywords": ["data augmentation", "datamart"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        # 'precondition': [],
        # 'hyperparms_to_tune': []

    })

    def __init__(self, *, hyperparams: QueryFromDataFrameHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False

    def _import_module(self):
        if self.hyperparams["url"].startswith('https://isi-datamart.edu'):
            from datamart import search, Dataset
            return 1
        if self.hyperparams["url"].startswith('https://datamart.d3m.vida-nyu.org'):
            from datamart_nyu import search, Dataset
            return 2
        return 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        status = self._import_module()
        if status == 0:
            print("not a valid url")
            return CallResult(DataFrame())
        if status == 1:
            # fixme one of the field
            res_list = search(
                query=self.hyperparams["query"], data=inputs)
        else:
            res_list = search(query=self.hyperparams["query"], data=inputs)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(res_list)

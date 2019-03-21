# input: dataframe, can be

import pandas as pd
import typing
import importlib
import logging


# importing d3m stuff
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from . import config



Inputs = DataFrame
Outputs = List
_logger = logging.getLogger(__name__)

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
        "python_path": "d3m.primitives.data_augmentation.datamart_query.DSBOX",
        "primitive_family": "DATA_AUGMENTATION",
        "description": "Queries datamart for available datasets. The JSON query specification is defined Datamart Query API. The primitive returns a list of ranked dataset metadata.",
        "algorithm_types": ["APPROXIMATE_DATA_AUGMENTATION"],
        "keywords": ["data augmentation", "datamart"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
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
            global ISI_datamart
            ISI_datamart = importlib.import_module('datamart')
            return 1
        if self.hyperparams["url"].startswith('https://datamart.d3m.vida-nyu.org'):
            global NYU_datamart
            NYU_datamart = importlib.import_module('datamart_nyu')
            # NYU_Dataset = importlib.import_module('datamart_nyu.Dataset')
            return 2
        return 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        status = self._import_module()
        if status == 0:
            _logger.info("not a valid url")
            return CallResult(None, True, 1)
        if status == 1:
            # fixme one of the field
            res_list = ISI_datamart.search(url=self.hyperparams["url"],
                query=self.hyperparams["query"], data=inputs)
        else:
            res_list = NYU_datamart.search(query=self.hyperparams["query"], data=inputs)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(res_list, True, 1)

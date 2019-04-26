import typing
import importlib
import logging
import frozendict
import copy
import pandas as pd
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
import collections
from datamart.entries_new import DatamartSearchResult

Inputs = container.Dataset
Outputs = container.Dataset
_logger = logging.getLogger(__name__)

class DatamartDownloadHyperparams(hyperparams.Hyperparams):
    # indexes of dataset to choose from
    #

    url = hyperparams.Hyperparameter[str](
        default='https://isi-datamart.edu',
        description='url indicates which datamart resource to use',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    search_result = hyperparams.Hyperparameter[dict](
        default=dict(),
        description="The list of serialized search result config",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    return_format = hyperparams.Enumeration(
        values=['ds','df'],
        default='ds',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="the return format, ds for dataset, df for dataframe",
    )

class DatamartDownload(TransformerPrimitiveBase[Inputs, Outputs, DatamartDownloadHyperparams]):
    '''
    A primitive that takes a list of datamart dataset and choose 1 or a few best dataframe and perform join, return an accessible d3m.dataframe for further processing
    '''
    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "datamart-download",
        "version": config.VERSION,
        "name": "Datamart Download",
        "python_path": "d3m.primitives.data_augmentation.datamart_download.DSBOX",
        "description": "Download the corresponding search result's dataset",
        "primitive_family": "DATA_AUGMENTATION",
        "algorithm_types": ["APPROXIMATE_DATA_AUGMENTATION"],  # fix me!
        "keywords": ["datamart", "download"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        # 'precondition': [],
        # 'hyperparams_to_tune': []

    })

    def __init__(self, *, hyperparams: DatamartDownloadHyperparams)-> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self.search_result = []
        self.search_result = DatamartSearchResult.construct(hyperparams["search_result"])
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
        input_dataset = inputs
        if status == 0:
            _logger.error("not a valid url")
            return CallResult(None, True, 1)
        if status == 1:  # run isi-datamart
            download_result = self.search_result.download(supplied_data=inputs, generate_metadata = True, return_format = self.hyperparams['return_format'])

        self._has_finished = True
        self._iterations_done = True
        return CallResult(download_result, True, 1)
import d3m.container
from d3m.metadata import hyperparams, params
from typing import Dict, List
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import random
import collections
import logging

from . import config

Inputs = d3m.container.DataFrame
Outputs = d3m.container.DataFrame
_logger = logging.getLogger(__name__)

class VoterHyperparameter(hyperparams.Hyperparams):
    classifier_voting_strategy = hyperparams.Enumeration(
        values=['random', 'majority'],
        default='majority',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="For classification problem, pick a prediction result if there are multiple results based on which strategy"
    )


class Voter(TransformerPrimitiveBase[Inputs, Outputs, VoterHyperparameter]):
    """
    Voter primitive
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-voter",
        "version": config.VERSION,
        "name": "ISI DSBox Prediction Voter",
        "description": "Voting primitive for choosing one prediction if there are multiple predictions",
        "python_path": "d3m.primitives.data_cleaning.voter.DSBOX",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["vote"],
        "installation": [config.INSTALLATION],
    })

    def __repr__(self):
        return "%s(%r)" % ('Voter', self.__dict__)

    def __init__(self, *, hyperparams: VoterHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._input_data: Inputs = None
        self._input_data_copy = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if 'd3mIndex' in list(inputs.columns.values) and len(set(inputs.loc[:, 'd3mIndex'].tolist())) < inputs.shape[0]:
            data_dict = dict()
            indices = list()
            for row in range(inputs.shape[0]):
                idx = inputs.iloc[row, :]['d3mIndex']
                if idx not in data_dict:
                    data_dict[idx] = list()
                    indices.append(idx)
                data_dict[idx].append(inputs.iloc[row, :].drop('d3mIndex').tolist())
            for key in data_dict:
                data_dict[key] = self._get_target(data_dict[key])
            new_df = inputs[0:0].drop('d3mIndex', axis=1)
            for idx in indices:
                new_df.loc[idx] = data_dict[idx]
            old_metadata = dict(new_df.metadata.query(()))
            old_metadata["dimension"] = dict(old_metadata["dimension"])
            old_metadata["dimension"]["length"] = new_df.shape[0]
            new_df.metadata = new_df.metadata.update((), old_metadata)
            return CallResult(new_df, True, 1)
        return CallResult(inputs, True, 1)

    def _get_target(self, lst):
        if self.hyperparams["classifier_voting_strategy"] == "random":
            return random.choice(lst)

        if self.hyperparams["classifier_voting_strategy"] == "majority":
            if len(lst) == 1:
                return lst[0]
            counts = collections.Counter([tuple(x) for x in lst])
            max_occurrence = max(counts.values())
            max_occurrence_elements = [x for x in lst if counts[tuple(x)] == max_occurrence]
            if len(max_occurrence_elements) == 1:
                return max_occurrence_elements[0]
            return random.choice(max_occurrence_elements)

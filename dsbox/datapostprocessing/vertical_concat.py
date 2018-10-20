from d3m import container, types
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams, params
from dsbox.datapreprocessing.cleaner import config
from d3m.primitive_interfaces.base import CallResult
import common_primitives.utils as common_utils
import pandas as pd
import warnings

__all__ = ('VerticalConcat',)

Inputs = container.List
Outputs = container.DataFrame


class VerticalConcatHyperparams(hyperparams.Hyperparams):
    ignore_index = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether new df should use original index or not"
    )
    sort_on_primary_key = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether new df will be sorted based on primary key, mostly time it will be d3mIndex"
    )


class VerticalConcat(TransformerPrimitiveBase[Inputs, Outputs, VerticalConcatHyperparams]):
    """
        A primitive which concat a list of dataframe to a single dataframe vertically
    """

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-vertical-concat",
        "version": config.VERSION,
        "name": "DSBox vertically concat",
        "description": "A primitive which concat a list of dataframe to a single dataframe vertically",
        "python_path": "d3m.primitives.dsbox.VerticalConcat",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["concat", "vertical"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: VerticalConcatHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._training_data = None
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        new_df = pd.concat([x for x in inputs], ignore_index=self.hyperparams["ignore_index"])
        if self.hyperparams["sort_on_primary_key"]:
            primary_key_col = common_utils.list_columns_with_semantic_types(metadata=new_df.metadata, semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/PrimaryKey"])
            if not primary_key_col:
                warnings.warn("No PrimaryKey column found. Will not sort on PrimaryKey")
                return CallResult(self._update_metadata(new_df))
            new_df = new_df.sort_values([new_df.columns[pos] for pos in primary_key_col])
        return CallResult(self._update_metadata(new_df))

    @staticmethod
    def _update_metadata(df: container.DataFrame) -> container.DataFrame:
        old_metadata = dict(df.metadata.query(()))
        old_metadata["dimension"] = dict(old_metadata["dimension"])
        old_metadata["dimension"]["length"] = df.shape[0]
        df.metadata = df.metadata.update((), old_metadata)
        return df

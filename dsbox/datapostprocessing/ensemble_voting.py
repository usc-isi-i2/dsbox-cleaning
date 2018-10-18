from d3m import container, types
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams, params
from dsbox.datapreprocessing.cleaner import config
from d3m.primitive_interfaces.base import CallResult
import common_primitives.utils as common_utils
from dsbox.datapostprocessing.vertical_concat import VerticalConcat, VerticalConcatHyperparams


__all__ = ('EnsembleVoting',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class EnsembleVotingHyperparams(hyperparams.Hyperparams):
    ensemble_method = hyperparams.Enumeration(
        values=['majority', 'mean', 'max', 'min', 'median', 'random'],
        default='majority',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls which ensemble method should be used",
    )


class EnsembleVoting(TransformerPrimitiveBase[Inputs, Outputs, EnsembleVotingHyperparams]):
    """
        A primitive which generate single prediction result for one index if there is many
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-ensemble-voting",
        "version": config.VERSION,
        "name": "DSBox ensemble voting",
        "description": "A primitive which generate single prediction result for one index if there is many",
        "python_path": "d3m.primitives.dsbox.EnsembleVoting",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["voting", "ensemble"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: EnsembleVotingHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._training_data = None
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        df = inputs.copy()
        index_col = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata, semantic_types=["https://metadata.datadrivendiscovery.org/types/PrimaryKey"])
        predict_target_col = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata, semantic_types=["https://metadata.datadrivendiscovery.org/types/PredictedTarget"])
        new_df = self._get_index_and_target_df(inputs=df, use_cols=index_col+predict_target_col)

        if self.hyperparams["ensemble_method"] == 'majority':
            groupby_df = new_df.groupby([new_df.columns[pos] for pos in index_col]).agg(
                lambda x: x.value_counts().index[0]).reset_index(drop=False)
            ret_df = container.DataFrame(groupby_df)
            ret_df.metadata = new_df.metadata

        if self.hyperparams["ensemble_method"] == 'max':
            groupby_df = new_df.groupby([new_df.columns[pos] for pos in index_col]).max().reset_index(drop=False)
            ret_df = container.DataFrame(groupby_df)
            ret_df.metadata = new_df.metadata

        if self.hyperparams["ensemble_method"] == 'min':
            groupby_df = new_df.groupby([new_df.columns[pos] for pos in index_col]).min().reset_index(drop=False)
            ret_df = container.DataFrame(groupby_df)
            ret_df.metadata = new_df.metadata

        return CallResult(self._update_metadata(df=ret_df))

    @staticmethod
    def _get_index_and_target_df(inputs: container.DataFrame, use_cols):
        metadata = common_utils.select_columns_metadata(inputs_metadata=inputs.metadata, columns=use_cols)
        new_df = inputs.iloc[:, use_cols]
        new_df.metadata = metadata
        return new_df

    @staticmethod
    def _update_metadata(df: container.DataFrame) -> container.DataFrame:
        old_metadata = dict(df.metadata.query(()))
        old_metadata["dimension"] = dict(old_metadata["dimension"])
        old_metadata["dimension"]["length"] = df.shape[0]
        df.metadata = df.metadata.update((), old_metadata)
        return df

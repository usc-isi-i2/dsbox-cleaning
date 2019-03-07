from d3m import container
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from dsbox.datapreprocessing.cleaner import config
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata.base import ALL_ELEMENTS
import common_primitives.utils as common_utils
import warnings

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
        "python_path": "d3m.primitives.data_preprocessing.ensemble_voting.DSBOX",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": ["ENSEMBLE_LEARNING"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["voting", "ensemble"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: EnsembleVotingHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        index_col = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata, semantic_types=["https://metadata.datadrivendiscovery.org/types/PrimaryKey"])
        if not index_col:
            warnings.warn("Did not find primary key column. Can not vote, output origin")
            return CallResult(inputs)

        predict_target_col = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata, semantic_types=["https://metadata.datadrivendiscovery.org/types/PredictedTarget"])
        if not index_col:
            warnings.warn("Did not find PredictedTarget column. Can not vote, output origin")
            return CallResult(inputs)

        df = inputs.copy()
        # temporary fix for index type problem
        # fix data type to be correct here
        for each_col in index_col:
            col_semantic_type = df.metadata.query((ALL_ELEMENTS, each_col))['semantic_types']
            if 'http://schema.org/Integer' in col_semantic_type and df[df.columns[each_col]].dtype == 'O':
                df[df.columns[each_col]] = df[df.columns[each_col]].astype(int)

        new_df = self._get_index_and_target_df(inputs=df, use_cols=index_col + predict_target_col)

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

from d3m import container
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from dsbox.datapreprocessing.cleaner import config
from d3m.primitive_interfaces.base import CallResult
import common_primitives.utils as common_utils
import d3m.metadata.base as mbase
import warnings

__all__ = ('Unfold',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class UnfoldHyperparams(hyperparams.Hyperparams):
    unfold_semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str]("str"),
        default=["https://metadata.datadrivendiscovery.org/types/PredictedTarget"],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=
        """
        A set of semantic types that the primitive will unfold.
        Only 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' by default.
        """,
    )
    use_pipeline_id_semantic_type = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=
        """
        Controls whether semantic_type will be used for finding pipeline id column in input dataframe.
        If true, it will look for 'https://metadata.datadrivendiscovery.org/types/PipelineId' for pipeline id column,
        and create attribute columns using header: attribute_{pipeline_id}. 
        eg. 'binaryClass_{a3180751-33aa-4790-9e70-c79672ce1278}'
        If false, create attribute columns using header: attribute_{0,1,2,...}.
        eg. 'binaryClass_0', 'binaryClass_1'
        """,
    )


class Unfold(TransformerPrimitiveBase[Inputs, Outputs, UnfoldHyperparams]):
    """
        A primitive which concat a list of dataframe to a single dataframe vertically
    """

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-unfold",
        "version": config.VERSION,
        "name": "DSBox unfold",
        "description": "A primitive which unfold a vertically concatenated dataframe",
        "python_path": "d3m.primitives.dsbox.Unfold",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["unfold"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: UnfoldHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._sorted_pipe_ids = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        primary_key_cols = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata,
            semantic_types=["https://metadata.datadrivendiscovery.org/types/PrimaryKey"]
        )

        unfold_cols = common_utils.list_columns_with_semantic_types(
            metadata=inputs.metadata,
            semantic_types=self.hyperparams["unfold_semantic_types"]
        )

        if not primary_key_cols:
            warnings.warn("Did not find primary key column for grouping. Will not unfold")
            return CallResult(inputs)

        if not unfold_cols:
            warnings.warn("Did not find any column to unfold. Will not unfold")
            return CallResult(inputs)

        primary_key_col_names = [inputs.columns[pos] for pos in primary_key_cols]
        unfold_col_names = [inputs.columns[pos] for pos in unfold_cols]

        if self.hyperparams["use_pipeline_id_semantic_type"]:
            pipeline_id_cols = common_utils.list_columns_with_semantic_types(
                metadata=inputs.metadata,
                semantic_types=["https://metadata.datadrivendiscovery.org/types/PipelineId"]
            )

            if len(pipeline_id_cols) >= 2:
                warnings.warn("Multiple pipeline id columns found. Will use first.")

            if pipeline_id_cols:
                inputs = inputs.sort_values(primary_key_col_names + [inputs.columns[pos] for pos in pipeline_id_cols])
                self._sorted_pipe_ids = sorted(inputs.iloc[:, pipeline_id_cols[0]].unique())
            else:
                warnings.warn(
                    "No pipeline id column found by 'https://metadata.datadrivendiscovery.org/types/PipelineId'")

        new_df = self._get_new_df(inputs=inputs, use_cols=primary_key_cols + unfold_cols)

        groupby_df = inputs.groupby(primary_key_col_names)[unfold_col_names].aggregate(
            lambda x: container.List(x)).reset_index(drop=False)

        ret_df = container.DataFrame(groupby_df)
        ret_df.metadata = new_df.metadata
        ret_df = self._update_metadata_dimension(df=ret_df)

        split_col_names = [inputs.columns[pos] for pos in unfold_cols]

        ret_df = self._split_aggregated(df=ret_df, split_col_names=split_col_names)
        ret_df = common_utils.remove_columns(
            inputs=ret_df,
            column_indices=[ret_df.columns.get_loc(name) for name in split_col_names]
        )

        return CallResult(ret_df)

    @staticmethod
    def _get_new_df(inputs: container.DataFrame, use_cols: list):
        metadata = common_utils.select_columns_metadata(inputs_metadata=inputs.metadata, columns=use_cols)
        new_df = inputs.iloc[:, use_cols]
        new_df.metadata = metadata
        return new_df

    @staticmethod
    def _update_metadata_dimension(df: container.DataFrame) -> container.DataFrame:
        old_metadata = dict(df.metadata.query(()))
        old_metadata["dimension"] = dict(old_metadata["dimension"])
        old_metadata["dimension"]["length"] = df.shape[0]
        df.metadata = df.metadata.update((), old_metadata)
        return df

    def _split_aggregated(self, df: container.DataFrame, split_col_names: list) -> container.DataFrame:
        lengths = [len(df.loc[0, col_name]) for col_name in split_col_names]

        for idx, col_name in enumerate(split_col_names):
            if self._sorted_pipe_ids:
                if len(self._sorted_pipe_ids) == lengths[idx]:
                    extend_col_names = ["{}_{}".format(col_name, i) for i in self._sorted_pipe_ids]
                else:
                    raise ValueError("Unique number of pipeline ids not equal to the number of aggregated values")
            else:
                extend_col_names = ["{}_{}".format(col_name, i) for i in range(lengths[idx])]

            extends = container.DataFrame(df.loc[:, col_name].values.tolist(), columns=extend_col_names)

            df = common_utils.horizontal_concat(left=df, right=extends)
            origin_metadata = dict(df.metadata.query((mbase.ALL_ELEMENTS, df.columns.get_loc(col_name))))

            for name in extend_col_names:
                col_idx = df.columns.get_loc(name)
                origin_metadata["name"] = name
                df.metadata = df.metadata.update((mbase.ALL_ELEMENTS, col_idx), origin_metadata)

        return df

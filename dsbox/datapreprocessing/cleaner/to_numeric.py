import logging
import typing

import numpy as np
import pandas as pd

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils

from . import config

__all__ = ('ToNumeric',)
_logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. The default is all columns.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    drop_non_numeric_columns = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="If True, drop all non-numeric columns",
    )


class ToNumeric(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which returns a DataFrame that is purely numeric. Numeric semantic type
    columns are converted to numerical structual type columns. Columns that are not numeric
    semantic type are droped. Missing values are encoded as NaN. Many SKLearn primitives
    require purely numeric DataFrame as input. It useful to run this primitive after
    running d3m.primitives.data_preprocessing.Encoder.DSBOX to encode categorical columns
    and d3m.primitives.dsbox.CorexText to encode text columns. This primitve preserves the
    D3M index column.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7ddf2fd8-2f7f-4e53-96a7-0d9f5aeecf93',
            'version': config.VERSION,
            'name': 'ISI DSBox To Numeric DataFrame',
            'desription': 'Convert to purely numeric DataFrame',
            'python_path': 'd3m.primitives.data_transformation.ToNumeric.DSBOX',
            'source': {
                'name': config.D3M_PERFORMER_TEAM,
                'contact': config.D3M_CONTACT,
                'uris': [config.REPOSITORY]
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    @classmethod
    def _can_use_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))
        semantic_type = column_metadata.get('semantic_types', None)

        if semantic_type is None:
            return False

        return 'http://schema.org/Integer' in semantic_type or 'http://schema.org/Float' in semantic_type

    @classmethod
    def _get_columns(cls, inputs_metadata: metadata_base.DataMetadata, hyperparams: hyperparams.Hyperparams) -> typing.Sequence[int]:
        def can_use_column(column_index: int) -> bool:
            return cls._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = utils.get_columns_to_use(inputs_metadata, hyperparams['use_columns'], hyperparams['exclude_columns'], can_use_column)
        return columns_to_use

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        columns_to_use = self._get_columns(inputs.metadata, self.hyperparams)
        _logger.debug(f'converting columns: {columns_to_use}')
        _logger.debug(f'converting columns: {inputs.iloc[:, columns_to_use].columns}')
        output = inputs.copy()
        for col in columns_to_use:
            output.iloc[:, col] = pd.to_numeric(output.iloc[:, col])
            column_metadata = output.metadata.query((metadata_base.ALL_ELEMENTS, col))
            semantic_type = column_metadata.get('semantic_types', None)
            if 'http://schema.org/Integer' in semantic_type:
                output.metadata = output.metadata.update((metadata_base.ALL_ELEMENTS, col), {'structural_type': int})
            elif 'http://schema.org/Float' in semantic_type:
                output.metadata = output.metadata.update((metadata_base.ALL_ELEMENTS, col), {'structural_type': float})
            # What to do with missing values?
            # has_missing_value = pd.isnull(output.iloc[:, col]).sum() > 0
        if self.hyperparams['drop_non_numeric_columns']:
            _logger.debug(f'dropping columns: {list(np.where(output.dtypes == object)[0])}')
            _logger.debug(f'dropping columns: {output.iloc[:, list(np.where(output.dtypes == object)[0])].columns}')
            # np.where returns int64 instead of int, D3M metadata checks for int
            numeric_colum_indices = [int(x) for x in np.where(output.dtypes != object)[0]]
            output = output.iloc[:, numeric_colum_indices]
            output.metadata = utils.select_columns_metadata(output.metadata, numeric_colum_indices)

        return base.CallResult(output)

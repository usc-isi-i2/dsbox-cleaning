import os
import typing
import logging

import pandas  # type: ignore

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from . import config

__all__ = ('DenormalizePrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset
_logger = logging.getLogger(__name__)

class DenormalizeHyperparams(hyperparams.Hyperparams):
    starting_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="From which resource to start denormalizing. If \"None\" then it starts from the dataset entry point.",
    )
    recursive = hyperparams.Hyperparameter[bool](
        True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize recursively?",
    )
    many_to_many = hyperparams.Hyperparameter[bool](
        True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize also many-to-many relations?",
    )


class Denormalize(transformer.TransformerPrimitiveBase[Inputs, Outputs, DenormalizeHyperparams]):
    """
    A primitive which converts a dataset with multiple tabular resources into a dataset with only one tabular resource,
    based on known relations between tabular resources. Any resource which can be joined is joined, and other resources
    are discarded.
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-denormalize(from d3m)",
        "version": config.VERSION,
        "name": "DSBox denormalize",
        "description": "Adapted from d3m.common_primitives",
        "python_path": "d3m.primitives.normalization.denormalize.DSBOX",
        "primitive_family": "NORMALIZATION",
        "algorithm_types": ["DATA_NORMALIZATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["NORMALIZATION", "Scaler"],
        "installation": [config.INSTALLATION]
    })


    # TODO: Implement support for M2M relations.
    # TODO: This should work recursively. If any resource being pulled brings more foreign key, they should be resolved as well. Without loops of course.
    # TODO: When copying metadata, copy also all individual metadata for columns and rows, and any recursive metadata for nested data.
    # TODO: Implement can_accept.
    # TODO: This should remove only resources which were joined to the main resource, and not all resources. Do we even want to remove other resources at all?
    # TODO: Add all column names together to "other names" metadata for column.

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # If only one resource is in the dataset, we do not have anything to do.
        if inputs.metadata.query(())['dimension']['length'] == 1:
            return base.CallResult(inputs)

        main_resource_id = self.hyperparams['starting_resource']

        if main_resource_id is None:
            for resource_id in inputs.keys():
                if 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint' in inputs.metadata.query((resource_id,)).get('semantic_types', []):
                    main_resource_id = resource_id
                    break

        if main_resource_id is None:
            raise ValueError("A Dataset with multiple resources without an entry point and no starting resource specified as a hyper-parameter.")

        main_data = inputs[main_resource_id]
        main_columns_length = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']

        # There is only one resource now.
        top_level_metadata = dict(inputs.metadata.query(()))
        top_level_metadata['dimension'] = dict(top_level_metadata['dimension'])
        top_level_metadata['dimension']['length'] = 1

        # !!! changed part: remove unloaded metadata to pass the check function
        metadata = inputs.metadata.clear(top_level_metadata, source=self).set_for_value(None, source=self)
        other_keys = [*inputs]
        other_keys.remove(main_resource_id)
        for each_key in other_keys:
            metadata = metadata.remove(selector = (each_key,),recursive = True)
        # changed finished
        
        #metadata = inputs.metadata.clear(top_level_metadata, source=self).set_for_value(None, source=self)

        # Resource is not anymore an entry point.
        entry_point_metadata = dict(inputs.metadata.query((main_resource_id,)))
        entry_point_metadata['semantic_types'] = [
            semantic_type for semantic_type in entry_point_metadata['semantic_types'] if semantic_type != 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
        ]
        metadata = metadata.update((main_resource_id,), entry_point_metadata, source=self)

        data = None

        for column_index in range(main_columns_length):
            column_metadata = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS, column_index))

            if 'foreign_key' not in column_metadata:
                # We just copy over data and metadata.
                data, metadata = self._add_column(main_resource_id, data, metadata, self._get_column(main_data, column_index), column_metadata)
            else:
                assert column_metadata['foreign_key']['type'] == 'COLUMN', column_metadata

                if 'column_index' in column_metadata['foreign_key']:
                    data, metadata = self._join_by_index(
                        main_resource_id, inputs, column_index, data, metadata, column_metadata['foreign_key']['resource_id'],
                        column_metadata['foreign_key']['column_index'],
                    )
                elif 'column_name' in column_metadata['foreign_key']:
                    data, metadata = self._join_by_name(
                        main_resource_id, inputs, column_index, data, metadata, column_metadata['foreign_key']['resource_id'],
                        column_metadata['foreign_key']['column_name'],
                    )
                else:
                    assert False, column_metadata

        resources = {}
        resources[main_resource_id] = data

        # Number of columns had changed.
        all_rows_metadata = dict(inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS)))
        all_rows_metadata['dimension'] = dict(all_rows_metadata['dimension'])
        all_rows_metadata['dimension']['length'] = data.shape[1]
        metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS), all_rows_metadata, for_value=resources, source=self)

        
        # !!! changed part: load all dataset to resources
        '''
        other_keys = [*inputs]
        other_keys.remove(main_resource_id)
        for each_key in other_keys:
            metadata = metadata.remove(selector = (each_key,),recursive = True, source = resources)
        '''
        '''
        # this change only works for d3m v2018.6.5, for v2018.7.10, even the "metadata.remove" will check the resouces and metadata relationship: so we have to load all data to the resources before check/remove
        # !!! changed part: remove unloaded metadata to pass the check function
        other_keys = [*inputs]
        other_keys.remove(main_resource_id)
        for each_key in other_keys:
            metadata = metadata.remove(selector = (each_key,),recursive = True, source = resources)
        # changed finished
        '''
        metadata.check(resources)

        dataset = container.Dataset(resources, metadata)

        return base.CallResult(dataset)


        


    def _join_by_name(self, main_resource_id: str, inputs: Inputs, inputs_column_index: int, data: typing.Optional[pandas.DataFrame],
                      metadata: metadata_base.DataMetadata, foreign_resource_id: str, foreign_column_name: str) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:
        for column_index in range(inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']):
            if inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('name', None) == foreign_column_name:
                return self._join_by_index(main_resource_id, inputs, inputs_column_index, data, metadata, foreign_resource_id, column_index)

        raise ValueError(
            "Cannot resolve foreign key with column name '{column_name}' in resource with ID '{resource_id}'.".format(
                resource_id=foreign_resource_id,
                column_name=foreign_column_name,
            ),
        )

    def _join_by_index(self, main_resource_id: str, inputs: Inputs, inputs_column_index: int, data: typing.Optional[pandas.DataFrame],
                       metadata: metadata_base.DataMetadata, foreign_resource_id: str, foreign_column_index: int) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:
        main_column_metadata = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS, inputs_column_index))

        main_data = inputs[main_resource_id]
        foreign_data = inputs[foreign_resource_id]

        value_to_index = {}
        for value_index, value in enumerate(foreign_data.iloc[:, foreign_column_index]):
            # TODO: Check if values are not unique.
            value_to_index[value] = value_index
        rows = []
        for value in main_data.iloc[:, inputs_column_index]:
            rows.append([foreign_data.iloc[value_to_index[value], j] for j in range(len(foreign_data.columns))])
        if data is None:
            data_columns_length = 0
        else:
            data_columns_length = data.shape[1]

        # Copy over metadata.
        foreign_data_columns_length = inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']
        for column_index in range(foreign_data_columns_length):
            column_metadata = dict(inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS, column_index)))

            # Foreign keys can reference same foreign row multiple times, so values in this column might not be even
            # unique anymore, nor they are a primary key at all. Sso we remove the semantic type marking a column as such.
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in column_metadata.get('semantic_types', []):
                column_metadata['semantic_types'] = [
                    semantic_type for semantic_type in column_metadata['semantic_types'] if semantic_type != 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
                ]

            # If the original index column was an attribute, make sure the new index column is as well.
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in main_column_metadata.get('semantic_types', []):
                if 'https://metadata.datadrivendiscovery.org/types/Attribute' not in column_metadata['semantic_types']:
                    column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/Attribute')

            # If the original index column was a suggested target, make sure the new index column is as well.
            if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in main_column_metadata.get('semantic_types', []):
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' not in column_metadata['semantic_types']:
                    column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

            metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS, data_columns_length + column_index), column_metadata, source=self)

        selected_data = pandas.DataFrame(rows)
        if data is None:
            data = selected_data
        else:
            #import pdb
            #pdb.set_trace()
            #data = data.reset_index().drop(columns=['index'])
            selected_data = selected_data.set_index(data.index)
            #selected_data = selected_data.reset_index().drop(columns=['index'])
            data = pandas.concat([data, selected_data], axis=1, ignore_index=True)
        return data, metadata

    def _get_column(self, data: pandas.DataFrame, column_index: int) -> pandas.DataFrame:
        return data.iloc[:, [column_index]]

    def _add_column(self, main_resource_id: str, data: pandas.DataFrame, metadata: metadata_base.DataMetadata, column_data: pandas.DataFrame,
                    column_metadata: typing.Dict) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:

        assert column_data.shape[1] == 1

        if data is None:
            data = column_data
        else:
            #import pdb
            #pdb.set_trace()
            #data = data.reset_index().drop(columns=['index'])
            column_data = column_data.set_index(data.index)
            #column_data = column_data.reset_index().drop(columns=['index'])
            data = pandas.concat([data, column_data], axis=1)
            '''
            data = data.reset_index().drop(columns=['index'])
            selected_data_key = column_data.columns
            for each_key in selected_data_key:
                data[each_key] = column_data[each_key]
            '''
        metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS, data.shape[1] - 1), column_metadata, source=self)

        return data, metadata

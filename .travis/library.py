import copy
import glob
import json
import logging
import numpy as np
import typing

from d3m import index
from d3m.container.dataset import SEMANTIC_TYPES
from d3m.metadata.problem import TaskType, TaskSubtype
from template import DSBoxTemplate
from template_steps import TemplateSteps

class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template",
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "taskType": TaskType.CLASSIFICATION.name,
            "runType": "classification",
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() +
                     TemplateSteps.dsbox_feature_selector("classification",
                                                          first_input='data',
                                                          second_input='target') +
                     [
                         {
                             "name": "model_step",
                             "runtime": {
                                 "cross_validation": 10,
                                 "stratified": True
                             },
                             "primitives": [
                                 {
                                     "primitive":
                                         "d3m.primitives.classification.random_forest.SKlearn",
                                     "hyperparameters":
                                         {
                                            'use_semantic_types': [True],
                                            'return_result': ['new'],
                                            'add_index_columns': [True],
                                            # 'bootstrap': [True, False],
                                            # 'max_depth': [15, 30, None],
                                            # 'min_samples_leaf': [1, 2, 4],
                                            # 'min_samples_split': [2, 5, 10],
                                            # 'max_features': ['auto', 'sqrt'],
                                            # 'n_estimators': [10, 50, 100],
                                         }
                                 }
                             ],
                             "inputs": ["feature_selector_step", "target"]
                         }
                     ]
        }


class DefaultTimeseriesCollectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_timeseries_collection_template",
            "taskType": TaskType.CLASSIFICATION.name,
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskSubtype.BINARY.name, TaskSubtype.MULTICLASS.name},
            "runType": "timeseries",
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "random_forest_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',
                                                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "random_forest_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }


class DefaultRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_regression_template",
            "taskSubtype": {TaskSubtype.UNIVARIATE.name, TaskSubtype.MULTIVARIATE.name},
            "taskType": TaskType.REGRESSION.name,
            "runType": "regression",
            # See TaskType, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING',
            # 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION',
            # 'REGRESSION', 'TIME_SERIES_FORECASTING', 'VERTEX_NOMINATION'
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "test_validation": 5,
                        "stratified": False
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 4, 5],
                                    'n_estimators': [100, 130, 165, 200],
                                    'learning_rate': [0.1, 0.23, 0.34, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }
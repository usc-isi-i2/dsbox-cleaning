import json

from collections import defaultdict

import d3m.metadata.base as mbase
import numpy as np
import pandas as pd
import pytypes
import logging
import traceback

from d3m import container, types
from d3m.metadata import hyperparams
from d3m.metadata.base import DataMetadata, PrimitiveFamily, PrimitiveAlgorithmType, Selector, ALL_ELEMENTS
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from dsbox.datapreprocessing.cleaner.dependencies.date_featurizer_org import DateFeaturizerOrg
from dsbox.datapreprocessing.cleaner.dependencies.spliter import PhoneParser, PunctuationParser, NumAlphaParser
from dsbox.datapreprocessing.cleaner.dependencies.helper_funcs import HelperFunction
from dsbox.datapreprocessing.cleaner.dependencies import category_detection, dtype_detector, \
    feature_compute_hih as fc_hih, feature_compute_lfh as fc_lfh
from . import config

# from . import date_detector


_logger = logging.getLogger(__name__)

Input = container.DataFrame
Output = container.DataFrame

VERBOSE = 0

computable_metafeatures = [
    'ratio_of_values_containing_numeric_char', 'ratio_of_numeric_values',
    'number_of_outlier_numeric_values', 'num_filename', 'number_of_tokens_containing_numeric_char',
    'number_of_numeric_values_equal_-1', 'most_common_numeric_tokens', 'most_common_tokens',
    'ratio_of_distinct_tokens', 'number_of_missing_values',
    'number_of_distinct_tokens_split_by_punctuation', 'number_of_distinct_tokens',
    'ratio_of_missing_values', 'semantic_types', 'number_of_numeric_values_equal_0',
    'number_of_positive_numeric_values', 'most_common_alphanumeric_tokens',
    'numeric_char_density', 'ratio_of_distinct_values', 'number_of_negative_numeric_values',
    'target_values', 'ratio_of_tokens_split_by_punctuation_containing_numeric_char',
    'ratio_of_values_with_leading_spaces', 'number_of_values_with_trailing_spaces',
    'ratio_of_values_with_trailing_spaces', 'number_of_numeric_values_equal_1',
    'natural_language_of_feature', 'most_common_punctuations', 'spearman_correlation_of_features',
    'number_of_values_with_leading_spaces', 'ratio_of_tokens_containing_numeric_char',
    'number_of_tokens_split_by_punctuation_containing_numeric_char', 'number_of_numeric_values',
    'ratio_of_distinct_tokens_split_by_punctuation', 'number_of_values_containing_numeric_char',
    'most_common_tokens_split_by_punctuation', 'number_of_distinct_values',
    'pearson_correlation_of_features',
    'semantic_types']

default_metafeatures = [
    'ratio_of_values_containing_numeric_char', 'ratio_of_numeric_values',
    'number_of_outlier_numeric_values', 'num_filename', 'number_of_tokens_containing_numeric_char', 'semantic_types']

metafeature_hyperparam = hyperparams.Enumeration(
    computable_metafeatures,
    computable_metafeatures[0],
    semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])


class Hyperparams(hyperparams.Hyperparams):
    metafeatures = hyperparams.Set(
        metafeature_hyperparam, default_metafeatures, min_size=1, max_size=len(computable_metafeatures),
        description="Compute metadata descriptions of the dataset",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])


class Profiler(TransformerPrimitiveBase[Input, Output, Hyperparams]):
    """
    data profiler moduel. Now only supports csv data.

    Parameters:
    ----------
    _punctuation_outlier_weight: a integer
        the coefficient used in outlier detection for punctuation. default is 3

    _numerical_outlier_weight

    _token_delimiter: a string
        delimiter that used to seperate tokens, default is blank space " ".

    _detect_language: boolean
        true: do detect language; false: not detect language

    _topk: a integer

    _verbose: boolean
        control the _verbose

    Attributes:
    ----------
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'b2612849-39e4-33ce-bfda-24f3e2cb1e93',
        'version': config.VERSION,
        'name': "DSBox Profiler",
        'description': 'Generate profiles of datasets',
        'python_path': 'd3m.primitives.dsbox.Profiler',
        'primitive_family': PrimitiveFamily.SCHEMA_DISCOVERY,
        'algorithm_types': [
            PrimitiveAlgorithmType.DATA_PROFILING,
        ],
        'keywords': ['data_profiler'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            'uris': [config.REPOSITORY],
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        # A metafeature about preconditions required for this primitive to operate well.
        "precondition": [],
        "hyperparms_to_tune": []
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        # All other attributes must be private with leading underscore
        self._punctuation_outlier_weight = 3
        self._numerical_outlier_weight = 3
        self._token_delimiter = " "
        self._detect_language = False
        self._topk = 10
        self._verbose = VERBOSE
        self._sample_df = None
        self._DateFeaturizer = None
        # list of specified features to compute
        self._specified_features = hyperparams["metafeatures"] if hyperparams else default_metafeatures

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        generate features for the input.
        Input:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray, container.matrix, container.List]
        Output:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray, container.matrix, container.List]
        """
        # Wrap as container, if needed
        if not pytypes.is_of_type(inputs, types.Container):
            if isinstance(inputs, pd.DataFrame):
                inputs = container.DataFrame(inputs)
            elif isinstance(inputs, np.matrix):
                inputs = container.matrix(inputs)
            elif isinstance(inputs, np.ndarray):
                inputs = container.ndarray(inputs)
            elif isinstance(inputs, list):
                inputs = container.List(inputs)
            else:
                # Inputs is not a container, and cannot be converted to a container.
                # Nothing to do, since cannot store the computed metadata.
                return CallResult(inputs)

        # calling the utility to detect integer and float datatype columns
        inputs = dtype_detector.detector(inputs)

        # calling the utility to categorical datatype columns
        metadata = self._produce(inputs, inputs.metadata, [])
        # I guess there are updating the metdata here
        inputs.metadata = metadata

        if inputs.shape[0] > 100:
            self._sample_df = inputs.dropna().iloc[0:100, :]
        else:
            self._sample_df = inputs

        # calling date detector

        self._DateFeaturizer = DateFeaturizerOrg(inputs)
        try:
            cols = self._DateFeaturizer.detect_date_columns(self._sample_df)
        except Exception as e:
            _logger.error(traceback.print_exc(e))
            cols = list()
        if cols:
            indices = [inputs.columns.get_loc(c) for c in cols if c in inputs.columns]
            for i in indices:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                temp_value = list(old_metadata["semantic_types"])
                if len(temp_value) >= 1:
                    # if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in old_metadata.get(
                    #         "semantic_types", []):
                    #     old_metadata["semantic_types"] = (
                    #         'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    #         'https://metadata.datadrivendiscovery.org/types/Attribute')
                    if 'https://metadata.datadrivendiscovery.org/types/Time' not in old_metadata.get("semantic_types",
                                                                                                     []):
                        old_metadata["semantic_types"] += ('https://metadata.datadrivendiscovery.org/types/Time',)
                if isinstance(self._sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(self._sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                _logger.info(
                    "Date detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
                    {
                        'column_index': i,
                        'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i))),
                        'new_metadata': old_metadata,
                    },
                )

                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        # calling the PhoneParser detector

        try:
            PhoneParser_indices = PhoneParser.detect(df=self._sample_df)
        except Exception as e:
            _logger.error(traceback.print_exc(e))
            PhoneParser_indices = dict()
        if PhoneParser_indices.get("columns_to_perform"):
            for i in PhoneParser_indices["columns_to_perform"]:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                # print("old metadata", old_metadata)
                if 'https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber' not in old_metadata.get(
                        "semantic_types", []):
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber',)
                if 'https://metadata.datadrivendiscovery.org/types/UnnormalizedEntity' not in old_metadata.get(
                        "semantic_types", []):
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/UnnormalizedEntity',)

                if isinstance(self._sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(self._sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                _logger.info(
                    "Phone detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
                    {
                        'column_index': i,
                        'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i))),
                        'new_metadata': old_metadata,
                    },
                )
                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        # calling the PunctuationSplitter detector

        try:
            PunctuationSplitter_indices = PunctuationParser.detect(df=self._sample_df)
        except Exception as e:
            _logger.error(traceback.print_exc(e))
            PunctuationSplitter_indices = dict()
        if PunctuationSplitter_indices.get("columns_to_perform"):
            for i in PunctuationSplitter_indices["columns_to_perform"]:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                if 'https://metadata.datadrivendiscovery.org/types/CanBeSplitByPunctuation' not in old_metadata.get(
                        "semantic_types", []):
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/CanBeSplitByPunctuation',)

                if isinstance(self._sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(self._sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                _logger.info(
                    "Punctuation detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
                    {
                        'column_index': i,
                        'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i))),
                        'new_metadata': old_metadata,
                    },
                )
                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        # calling the NumAlphaSplitter detector

        try:
            NumAlphaSplitter_indices = NumAlphaParser.detect(df=self._sample_df)
        except Exception as e:
            _logger.error(traceback.print_exc(e))
            NumAlphaSplitter_indices = dict()

        if NumAlphaSplitter_indices.get("columns_to_perform"):
            for i in NumAlphaSplitter_indices["columns_to_perform"]:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                if 'https://metadata.datadrivendiscovery.org/types/CanBeSplitByAlphanumeric' not in old_metadata.get(
                        "semantic_types", []):
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/CanBeSplitByAlphanumeric',)

                if isinstance(self._sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(self._sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                _logger.info(
                    "NumAlpha detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
                    {
                        'column_index': i,
                        'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i))),
                        'new_metadata': old_metadata,
                    },
                )
                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        inputs = self._relabel_categorical(inputs)
        return CallResult(inputs)

    @staticmethod
    def _relabel_categorical(inputs: Input) -> Output:
        for col in range(inputs.shape[1]):
            old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, col)))
            semantic_type = old_metadata.get('semantic_types', [])

            if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_type:
                if not HelperFunction.is_categorical(inputs.iloc[:, col]):
                    old_metadata['semantic_types'] = tuple(i for i in old_metadata['semantic_types'] if
                                                           i != 'https://metadata.datadrivendiscovery.org/types/CategoricalData')

                    numerics = pd.to_numeric(inputs.iloc[:, col], errors='coerce')
                    length = numerics.shape[0]
                    nans = numerics.isnull().sum()

                    if nans / length > 0.9:
                        if "http://schema.org/Text" not in old_metadata['semantic_types']:
                            old_metadata['semantic_types'] += ("http://schema.org/Text",)

                    else:
                        intcheck = (numerics % 1) == 0
                        if np.sum(intcheck) / length > 0.9:
                            if "http://schema.org/Integer" not in old_metadata['semantic_types']:
                                old_metadata['semantic_types'] += ("http://schema.org/Integer",)
                                old_metadata['structural_type'] = type(10)
                                inputs.iloc[:, col] = numerics
                        else:
                            if "http://schema.org/Float" not in old_metadata['semantic_types']:
                                old_metadata['semantic_types'] = ("http://schema.org/Float",)
                                old_metadata['structural_type'] = type(10.2)
                                inputs.iloc[:, col] = numerics

            inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, col), old_metadata)

        return inputs

    def _produce(self, inputs: Input, metadata: DataMetadata = None, prefix: Selector = None) -> DataMetadata:
        """
        Parameters:
        -----------
        Input:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray, container.matrix, container.List]
        metadata: DataMetadata
            Store generate metadata. If metadata is None, then inputs must be container, which has a metadata field to store the generated data.
        prefix: Selector
            Selector prefix into metadata

        """
        if isinstance(inputs, container.Dataset):
            for table_id, resource in inputs.items():
                prefix = prefix + [table_id]
                metadata = self._produce(resource, metadata, prefix)
        elif isinstance(inputs, list):
            for index, item in enumerate(inputs):
                metadata = self._produce(item, metadata, prefix + [index])
        elif isinstance(inputs, pd.DataFrame):
            metadata = self._profile_data(inputs, metadata, prefix)
        elif isinstance(inputs, np.matrix) or (isinstance(inputs, np.ndarray) and len(inputs.shape) == 2):
            df = pd.DataFrame(inputs)
            metadata = self._profile_data(df, metadata, prefix)
        elif isinstance(inputs, container.ndarray):
            metadata = self._profile_ndarray(inputs, metadata, prefix)

        return metadata

    def _profile_ndarray(self, array, metadata, prefix):
        # TODO: What to do with ndarrays?
        return metadata

    def _profile_data(self, data, metadata, prefix):

        """
        Main function to profile the data. This functions will
        1. calculate features
        2. update metadata with features

        Parameters
        ----------
        data: pandas.DataFrame that needs to be profiled
        ----------
        """
        if self._verbose:
            print("====================have a look on the data: ====================\n")
            print(data.head(2))

        # calculations
        if self._verbose:
            print("====================calculating the features ... ====================\n")

        # STEP 1: data-level calculations
        if ("pearson_correlation_of_features" in self._specified_features):
            corr_pearson = data.corr()
            corr_columns = list(corr_pearson.columns)
            corr_id = [data.columns.get_loc(n) for n in corr_columns]

        if ("spearman_correlation_of_features" in self._specified_features):
            corr_spearman = data.corr(method='spearman')
            corr_columns = list(corr_spearman.columns)
            corr_id = [data.columns.get_loc(n) for n in corr_columns]

        is_category = category_detection.category_detect(data)

        # STEP 2: column-level calculations
        column_counter = -1
        for column_name in data:
            column_counter += 1
            col = data[column_name]
            # dict: map feature name to content
            each_res = defaultdict(lambda: defaultdict())

            # if block updated on 6/26

            # old_metadata = dict(data.metadata.query((mbase.ALL_ELEMENTS, column_counter))),

            if 'semantic_types' in self._specified_features and is_category[column_name]:
                # rewrites old metadata
                old_metadata = dict(data.metadata.query((mbase.ALL_ELEMENTS, column_counter)))
                temp_value = list(old_metadata["semantic_types"])
                if len(temp_value) == 2:
                    ##print("$$$$$$", ('https://metadata.datadrivendiscovery.org/types/CategoricalData', temp_value[1]))
                    each_res["semantic_types"] = (
                        'https://metadata.datadrivendiscovery.org/types/CategoricalData', temp_value[-1])
                elif len(temp_value) == 1:
                    each_res["semantic_types"] = (
                        'https://metadata.datadrivendiscovery.org/types/CategoricalData', temp_value[-1])
                elif len(temp_value) == 3:
                    each_res["semantic_types"] = (
                        'https://metadata.datadrivendiscovery.org/types/CategoricalData', temp_value[-2],
                        temp_value[-1])

            if (("spearman_correlation_of_features" in self._specified_features) and
                    (column_name in corr_columns)):
                stats_sp = corr_spearman[column_name].describe()
                each_res["spearman_correlation_of_features"] = {'min': stats_sp['min'],
                                                                'max': stats_sp['max'],
                                                                'mean': stats_sp['mean'],
                                                                'median': stats_sp['50%'],
                                                                'std': stats_sp['std']}

            if (("spearman_correlation_of_features" in self._specified_features) and
                    (column_name in corr_columns)):
                stats_pr = corr_pearson[column_name].describe()
                each_res["pearson_correlation_of_features"] = {'min': stats_pr['min'],
                                                               'max': stats_pr['max'],
                                                               'mean': stats_pr['mean'],
                                                               'median': stats_pr['50%'],
                                                               'std': stats_pr['std']}

            if col.dtype.kind in np.typecodes['AllInteger'] + 'uMmf':
                if ("number_of_missing_values" in self._specified_features):
                    each_res["number_of_missing_values"] = pd.isnull(col).sum()
                if ("ratio_of_missing_values" in self._specified_features):
                    each_res["ratio_of_missing_values"] = pd.isnull(col).sum() / col.size
                if ("number_of_distinct_values" in self._specified_features):
                    each_res["number_of_distinct_values"] = col.nunique()
                if ("ratio_of_distinct_values" in self._specified_features):
                    each_res["ratio_of_distinct_values"] = col.nunique() / float(col.size)

            if col.dtype.kind == 'b':
                if ("most_common_raw_values" in self._specified_features):
                    fc_hih.compute_common_values(col.dropna().astype(str), each_res, self._topk)

            elif col.dtype.kind in np.typecodes['AllInteger'] + 'uf':
                fc_hih.compute_numerics(col, each_res,
                                        self._specified_features)  # TODO: do the checks inside the function
                if ("most_common_raw_values" in self._specified_features):
                    fc_hih.compute_common_values(col.dropna().astype(str), each_res, self._topk)

            else:

                # Need to compute str missing values before fillna
                if "number_of_missing_values" in self._specified_features:
                    each_res["number_of_missing_values"] = pd.isnull(col).sum()
                if "ratio_of_missing_values" in self._specified_features:
                    each_res["ratio_of_missing_values"] = pd.isnull(col).sum() / col.size

                col = col.astype(object).fillna('').astype(str)

                # compute_missing_space Must be put as the first one because it may change the data content, see function def for details
                fc_lfh.compute_missing_space(col, each_res, self._specified_features)
                # fc_lfh.compute_filename(col, each_res)
                fc_lfh.compute_length_distinct(col, each_res, delimiter=self._token_delimiter,
                                               feature_list=self._specified_features)
                if ("natural_language_of_feature" in self._specified_features):
                    fc_lfh.compute_lang(col, each_res)
                if ("most_common_punctuations" in self._specified_features):
                    fc_lfh.compute_punctuation(col, each_res, weight_outlier=self._punctuation_outlier_weight)

                fc_hih.compute_numerics(col, each_res, self._specified_features)
                if ("most_common_numeric_tokens" in self._specified_features):
                    fc_hih.compute_common_numeric_tokens(col, each_res, self._topk)
                if ("most_common_alphanumeric_tokens" in self._specified_features):
                    fc_hih.compute_common_alphanumeric_tokens(col, each_res, self._topk)
                if ("most_common_raw_values" in self._specified_features):
                    fc_hih.compute_common_values(col, each_res, self._topk)
                fc_hih.compute_common_tokens(col, each_res, self._topk, self._specified_features)
                if ("numeric_char_density" in self._specified_features):
                    fc_hih.compute_numeric_density(col, each_res)
                fc_hih.compute_contain_numeric_values(col, each_res, self._specified_features)
                fc_hih.compute_common_tokens_by_puncs(col, each_res, self._topk, self._specified_features)

            # update metadata for a specific column

            metadata = metadata.update(prefix + [ALL_ELEMENTS, column_counter], each_res)

            # _logger.info(
            #     "category detector. 'column_index': '%(column_index)d', 'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
            #     {
            #         'column_index': column_counter,
            #         'old_metadata': old_metadata,
            #         'new_metadata': dict(data.metadata.query((mbase.ALL_ELEMENTS, column_counter))),
            #     },
            # )
        return metadata


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

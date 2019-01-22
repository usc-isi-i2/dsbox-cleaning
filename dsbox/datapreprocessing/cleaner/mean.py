import logging
import typing

import pandas as pd  # type: ignore
import numpy as np

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from d3m.primitive_interfaces.base import CallResult
import stopit  # type: ignore

from d3m import container
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import UniformBool
import d3m.metadata.base as mbase
import common_primitives.utils as utils

from . import config

import common_primitives.utils as common_utils

Input = container.DataFrame
Output = container.DataFrame

_logger = logging.getLogger(__name__)


# store the mean value for each column in training data
class Params(params.Params):
    mean_values: typing.Dict
    type_columns: typing.Dict
    fitted: bool


class MeanHyperparameter(hyperparams.Hyperparams):
    verbose = UniformBool(default=False,
                          semantic_types=['http://schema.org/Boolean',
                                          'https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


class MeanImputation(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, MeanHyperparameter]):
    """
    Impute missing values using the `mean` value of the attribute.
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "7894b699-61e9-3a50-ac9f-9bc510466667",
        "version": config.VERSION,
        "name": "DSBox Mean Imputer",
        "description": "Impute missing values using the `mean` value of the attribute.",
        "python_path": "d3m.primitives.dsbox.MeanImputation",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["IMPUTATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": ["preprocessing", "imputation", "mean"],
        "installation": [config.INSTALLATION],
        "location_uris": [],
        "precondition": [hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES],
        # "effects": [hyperparams.base.PrimitiveEffects.NO_MISSING_VALUES],
        "hyperparms_to_tune": []
    })

    def __init__(self, *, hyperparams: MeanHyperparameter) -> None:

        super().__init__(hyperparams=hyperparams)
        # All primitives must define these attributes
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._train_x = None
        self._is_fitted = False
        self._has_finished = False
        self._iterations_done = False
        self._numeric_columns: typing.List = []
        self._categoric_columns: typing.List = []
        self._verbose = hyperparams['verbose'] if hyperparams else False

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['fitted']
        self._has_finished = self._is_fitted
        self._iterations_done = self._is_fitted
        self.mean_values = params['mean_values']
        self._numeric_columns = params['type_columns']['numeric_columns']
        self._categoric_columns = params['type_columns']['categoric_columns']

    def get_params(self) -> Params:
        return Params(
            mean_values=self.mean_values,
            type_columns={
                'numeric_columns': self._numeric_columns,
                'categoric_columns': self._categoric_columns
            },
            fitted=self._is_fitted)

    def set_training_data(self, *, inputs: Input) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Input
            The inputs.
        """
        attribute = utils.list_columns_with_semantic_types(
            inputs.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])
        nan_sum = 0
        for col in attribute:
            if str(inputs.dtypes[inputs.columns[col]]) != "object":
                nan_sum += inputs.iloc[:, col].isnull().sum()
            else:
                for i in range(inputs.shape[0]):
                    if inputs.iloc[i, col] == "" or pd.isnull(inputs.iloc[i, col]):
                        nan_sum += 1
        if nan_sum == 0:  # no missing value exists
            if self._verbose:
                print("Warning: no missing value in train dataset")
                _logger.info('no missing value in train dataset')

        self._train_x = inputs
        self._is_fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        get the mean value of each columns

        Parameters:
        ----------
        data: pandas dataframe
        """

        # if already fitted on current dataset, do nothing
        if self._is_fitted:
            return CallResult(None, self._has_finished, self._iterations_done)

        if (timeout is None):
            timeout = 2 ** 31 - 1

        if (iterations is None):
            self._iterations_done = True

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start fitting
            if self._verbose: print("=========> mean imputation method:")
            self.__get_fitted()

        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._iterations_done = True
            self._has_finished = True
        else:
            self._is_fitted = False
            self._iterations_done = False
            self._has_finished = False

        _logger.debug('Fit is_fitted %s', str(self._is_fitted))
        return CallResult(None, self._has_finished, self._iterations_done)

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        precond: run fit() before

        Parameters:
        ----------
        data: pandas dataframe
        """

        if (not self._is_fitted):
            # todo: specify a NotFittedError, like in sklearn
            raise ValueError("Calling produce before fitting.")

        # if (pd.isnull(inputs).sum().sum() == 0):    # no missing value exists
        #     if self._verbose: print ("Warning: no missing value in test dataset")
        #     self._has_finished = True
        #     return CallResult(inputs, self._has_finished, self._iterations_done)

        if (timeout is None):
            timeout = 2 ** 31 - 1

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        else:
            data = inputs[0].copy()

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose: print("=========> impute by mean value of the attribute:")

            data.iloc[:, self._numeric_columns] = data.iloc[:, self._numeric_columns].apply(
                lambda col: pd.to_numeric(col, errors='coerce'))

            # assume the features of testing data are same with the training data
            # therefore, only use the mean_values to impute, should get a clean dataset
            attribute = utils.list_columns_with_semantic_types(
                data.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])
            for col in attribute:
                if str(inputs.dtypes[inputs.columns[col]]) != "object":
                    data.iloc[:, col].fillna(self.mean_values[data.columns[col]], inplace=True)
                else:
                    for i in range(data.shape[0]):
                        if data.iloc[i, col] == "" or pd.isnull(data.iloc[i, col]):
                            data.iloc[i, col] = self.mean_values[data.columns[col]]
            data_clean = data

            # Update metadata
            for col in self._numeric_columns:
                old_metadata = dict(data_clean.metadata.query((mbase.ALL_ELEMENTS, col)))
                dtype = data_clean.iloc[:, col].dtype
                if str(dtype).lower().startswith("int"):
                    if "http://schema.org/Integer" not in old_metadata['semantic_types']:
                        old_metadata['semantic_types'] += ("http://schema.org/Integer",)
                    old_metadata["structural_type"] = type(10)
                elif str(dtype).lower().startswith("float"):
                    if "http://schema.org/Float" not in old_metadata['semantic_types']:
                        old_metadata['semantic_types'] += ("http://schema.org/Float",)
                    old_metadata["structural_type"] = type(10.2)

                data_clean.metadata = data_clean.metadata.update((mbase.ALL_ELEMENTS, col), old_metadata)

        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._has_finished = True
            self._iterations_done = True
            value = data_clean
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            _logger.warn('Produce timed out')
            self._has_finished = False
            self._iterations_done = False
        return CallResult(value, self._has_finished, self._iterations_done)

    @classmethod
    def _get_columns_to_fit(cls, inputs: Input, hyperparams: MeanHyperparameter):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                                     use_columns=hyperparams[
                                                                                         'use_columns'],
                                                                                     exclude_columns=hyperparams[
                                                                                         'exclude_columns'],
                                                                                     can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int,
                            hyperparams: MeanHyperparameter) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    def __get_fitted(self):
        attribute = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])

        # Mean for numerical columns

        self._numeric_columns = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        self._numeric_columns = [x for x in self._numeric_columns if x in attribute]

        _logger.debug('numeric columns %s', str(self._numeric_columns))

        # Convert selected columns to_numeric, then compute column mean, then convert to_dict
        self.mean_values = self._train_x.iloc[:, self._numeric_columns].apply(
            lambda col: pd.to_numeric(col, errors='coerce')).mean(axis=0).to_dict()

        for name in self.mean_values.keys():
            if pd.isnull(self.mean_values[name]):
                self.mean_values[name] = 0.0

        # Mode for categorical columns
        self._categoric_columns = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                     'http://schema.org/Boolean'])
        self._categoric_columns = [x for x in self._categoric_columns if x in attribute]

        _logger.debug('categorical columns %s', str(self._categoric_columns))

        mode_values = self._train_x.iloc[:, self._categoric_columns].mode(axis=0).iloc[0].to_dict()
        for name in mode_values.keys():
            if pd.isnull(mode_values[name]):
                # mode is nan
                rest = self._train_x[name].dropna()
                if rest.shape[0] == 0:
                    # every value is nan
                    mode = 0
                else:
                    mode = rest.mode().iloc[0]
                mode_values[name] = mode
        self.mean_values.update(mode_values)

        if self._verbose:
            import pprint
            print('mean imputation:')
            pprint.pprint(self.mean_values)

        _logger.debug('Mean values:')
        for name, value in self.mean_values.items():
            _logger.debug('  %s %s', name, str(value))

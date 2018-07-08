import logging
import typing

import pandas as pd #  type: ignore

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from d3m.primitive_interfaces.base import CallResult
import stopit #  type: ignore


from d3m import container
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import UniformBool
import common_primitives.utils as utils

from . import config

Input = container.DataFrame
Output = container.DataFrame

_logger = logging.getLogger(__name__)

# store the mean value for each column in training data
class Params(params.Params):
    mean_values : typing.Dict
    fitted : bool

class MeanHyperparameter(hyperparams.Hyperparams):
    verbose = UniformBool(default=False,
                          semantic_types=['http://schema.org/Boolean',
                                          'https://metadata.datadrivendiscovery.org/types/ControlParameter'])


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
        "algorithm_types": [ "IMPUTATION" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [ config.REPOSITORY ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "preprocessing", "imputation", "mean" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "precondition":  [hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES ],
        "effects": [ hyperparams.base.PrimitiveEffects.NO_MISSING_VALUES ],
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
        self._verbose = hyperparams['verbose'] if hyperparams else False


    def set_params(self, *, params: Params) -> None:
        self._is_fitted = params['fitted']
        self._has_finished = self._is_fitted
        self._iterations_done = self._is_fitted
        self.mean_values = params['mean_values']

    def get_params(self) -> Params:
        return Params(mean_values=self.mean_values, fitted=self._is_fitted)

    def set_training_data(self, *, inputs: Input) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Input
            The inputs.
        """
        if (pd.isnull(inputs).sum().sum() == 0):    # no missing value exists
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
            timeout = 2**31-1
        if (iterations is None):
            self._iterations_done = True

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start fitting
            if self._verbose : print("=========> mean imputation method:")
            self.__get_fitted()

        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._iterations_done = True
            self._has_finished = True
        else:
            import pdb
            pdb.set_trace()
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
        if (pd.isnull(inputs).sum().sum() == 0):    # no missing value exists
            if self._verbose: print ("Warning: no missing value in test dataset")
            self._has_finished = True
            return CallResult(inputs, self._has_finished, self._iterations_done)

        if (timeout is None):
            timeout = 2**31-1

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        else:
            data = inputs[0].copy()

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose: print("=========> impute by mean value of the attribute:")

            # assume the features of testing data are same with the training data
            # therefore, only use the mean_values to impute, should get a clean dataset
            data_clean = data.fillna(value=self.mean_values)


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

    def __get_fitted(self):
        attribute = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])

        # Mean for numerical columns
        numeric = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        numeric = [x for x in numeric if x in attribute]

        _logger.debug('numeric columns %s', str(numeric))

        self.mean_values = self._train_x.iloc[:, numeric].astype(float).mean(axis=0).to_dict()
        for name in self.mean_values.keys():
            if pd.isnull(self.mean_values[name]):
                self.mean_values[name] = 0.0

        # Mode for categorical columns
        categoric = utils.list_columns_with_semantic_types(
            self._train_x.metadata, ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                     'http://schema.org/Boolean'])
        categoric = [x for x in categoric if x in attribute]

        _logger.debug('categorical columns %s', str(categoric))

        mode_values = self._train_x.iloc[:, categoric].mode(axis=0).iloc[0].to_dict()
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

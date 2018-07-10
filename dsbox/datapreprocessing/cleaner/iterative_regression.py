import numpy as np
import pandas as pd
from . import missing_value_pred as mvp

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import stopit
import math
import typing

from d3m import container
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import UniformBool
import common_primitives.utils as utils

from . import config

Input = container.DataFrame
Output = container.DataFrame

# store the regression models for each missing-value column in training data
class Params(params.Params):
    regression_models: typing.Dict

class IterativeRegressionHyperparameter(hyperparams.Hyperparams):
    verbose = UniformBool(default=False,
                          semantic_types=['http://schema.org/Boolean',
                                          'https://metadata.datadrivendiscovery.org/types/ControlParameter'])


class IterativeRegressionImputation(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, IterativeRegressionHyperparameter]):
    """
    Impute the missing value by iteratively regress using other attributes.
        It will fit and fill the missing value in the training set, and store the learned models.
        In the `produce` phase, it will use the learned models to iteratively regress on the
        testing data again, and return the imputed testing data.

    Parameters:
    ----------
    verbose: bool
        Control the verbosity

    Attributes:
    ----------
    best_imputation: dict. key: column name; value: trained imputation method (parameters)
        could be sklearn regression model, or "mean" (which means the regression failed)

    """
    metadata = hyperparams.base.PrimitiveMetadata({
        ### Required
        "id": "f70b2324-1102-35f7-aaf6-7cd8e860acc4",
        "version": config.VERSION,
        "name": "DSBox Iterative Regression Imputer",
        "description": "Impute missing values using iterative regression",
        "python_path": "d3m.primitives.dsbox.IterativeRegressionImputation",
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
        "keywords": [ "preprocessing", "imputation" ],
        "installation": [ config.INSTALLATION ],
        "location_uris": [],
        "precondition": [ hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES ],
        "effects": [ hyperparams.base.PrimitiveEffects.NO_MISSING_VALUES ],
        "hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: IterativeRegressionHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)

        # All primitives must define these attributes
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._best_imputation : typing.Dict = {} # in params.regression_models
        self._train_x : Input = None
        self._is_fitted = True
        self._has_finished = True
        self._iterations_done = True
        self._verbose = hyperparams['verbose'] if hyperparams else False


    def set_params(self, *, params: Params) -> None:
        self._is_fitted = len(params['regression_models']) > 0
        self._has_finished = self._is_fitted
        self._best_imputation = params['regression_models']

    def get_params(self) -> Params:
        if self._is_fitted:
            return Params(regression_models=self._best_imputation)
        else:
            return Params(regression_models=dict())


    def set_training_data(self, *, inputs: Input) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Input
            The inputs.
        """
        if (pd.isnull(inputs).sum().sum() == 0):    # no missing value exists
            if self._verbose: print ("Warning: no missing value in train dataset")

        self._train_x = inputs
        self._is_fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        train imputation parameters. Now support:
        -> greedySearch

        for the method that not trainable, do nothing:
        -> interatively regression
        -> other

        Parameters:
        ----------
        data: pandas dataframe
        label: pandas series, used for the trainable methods
        """

        # if already fitted on current dataset, do nothing
        if self._is_fitted:
            return CallResult(None, self._has_finished, self._iterations_done)

        if (timeout is None):
            timeout = 2**31-1
        if (iterations is None):
            self._iterations_done = True
            iterations = 30

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            data = self._train_x.copy()

            # start fitting
            if self._verbose: print("=========> iteratively regress method:")
            data_clean, self._best_imputation = self.__iterativeRegress(data, iterations)

        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._iterations_done = True
            self._has_finished = True
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            self._is_fitted = False
            self._iterations_done = False
            self._has_finished = False
        return CallResult(None, self._has_finished, self._iterations_done)


    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        precond: run fit() before

        to complete the data, based on the learned parameters, support:
        -> greedy search

        also support the untrainable methods:
        -> iteratively regression
        -> other

        Parameters:
        ----------
        data: pandas dataframe
        label: pandas series, used for the evaluation of imputation

        TODO:
        ----------
        1. add evaluation part for __simpleImpute()

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
        if (iterations is None):
            self._iterations_done = True
            iterations = 30 # only works for iteratively_regre method

        data = inputs.copy()
        # record keys:
        keys = data.keys()
        index = data.index

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose: print("=========> iteratively regress method:")
            data_clean = self.__regressImpute(data, self._best_imputation, iterations)

        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._has_finished = True
            value = pd.DataFrame(data_clean, index, keys)
            value = container.DataFrame(value)
            value.metadata = data.metadata
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            print ("Timed Out...")
            self._is_fitted = False
            self._has_finished = False
            self._iterations_done = False
        return CallResult(value, self._has_finished, self._iterations_done)



    #============================================ helper functions ============================================
    def __iterativeRegress(self, data, iterations):
        '''
        init with simple imputation, then apply regression to impute iteratively
        '''
        # for now, cancel the evaluation part for iterativeRegress
        # is_eval = False
        # if (label_col_name==None or len(label_col_name)==0):
        #     is_eval = False
        # else:
        #     is_eval = True

        # indices for numeric attribute columns only
        attribute = utils.list_columns_with_semantic_types(
            data.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])
        numeric = utils.list_columns_with_semantic_types(
            data.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        numeric = [x for x in numeric if x in attribute]

        keys = data.keys()
        missing_col_id = []
        data = mvp.df2np(data, missing_col_id, self._verbose)

        # Impute numerical attributes only
        missing_col_id = [x for x in missing_col_id if x in numeric]
        missing_col_data = data[:, missing_col_id]

        # If all values in a column are missing, set that column to zero
        all_missing = np.sum(np.isnan(missing_col_data), axis=0)==missing_col_data.shape[0]
        for col, col_missing in enumerate(all_missing):
            if col_missing:
                missing_col_data[:,col] = 0

        imputed_data = np.zeros([data.shape[0], len(missing_col_id)])
        imputed_data_lastIter = missing_col_data
        # coeff_matrix = np.zeros([len(missing_col_id), data.shape[1]-1]) #coefficient vector for each missing value column
        model_list = [None]*len(missing_col_id)     # store the regression model
        epoch = iterations
        counter = 0
        # mean init all missing-value columns
        init_imputation = ["mean"] * len(missing_col_id)
        next_data = mvp.imputeData(data, missing_col_id, init_imputation, self._verbose)

        while (counter < epoch):
            for i in range(len(missing_col_id)):
                target_col = missing_col_id[i]
                next_data[:, target_col] = missing_col_data[:,i] #recover the column that to be imputed

                data_clean, model_list[i] = mvp.bayeImpute(next_data, target_col, self._verbose)
                next_data[:,target_col] = data_clean[:,target_col]    # update bayesian imputed column
                imputed_data[:,i] = data_clean[:,target_col]    # add the imputed data

                # if (is_eval):
                #     self.__evaluation(data_clean, label)

            # if (counter > 0):
            #     distance = np.square(imputed_data - imputed_data_lastIter).sum()
            #     if self._verbose: print("changed distance: {}".format(distance))
            imputed_data_lastIter = np.copy(imputed_data)
            counter += 1

        data[:,missing_col_id] = imputed_data_lastIter

        # convert model_list to dict
        model_dict = {}
        for i in range(len(model_list)):
            model_dict[keys[missing_col_id[i]]] = model_list[i]

        return data, model_dict

    def __regressImpute(self, data, model_dict, iterations):
        """
        """
        col_names = data.keys()
        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data = mvp.df2np(data, missing_col_id, self._verbose)


        model_list = [] # the model list
        new_missing_col_id = [] # the columns that have correspoding model
        # mask = np.ones((data.shape[1]), dtype=bool)   # false means: this column cannot be bring into impute
        # offset = 0  # offset from missing_col_id to new_missing_col_id

        for i in range(len(missing_col_id)):
            name = col_names[missing_col_id[i]]
            # if there is a column that not appears in trained model, impute it as "mean"
            if (name not in model_dict.keys()):
                data = mvp.imputeData(data, [missing_col_id[i]], ["mean"], self._verbose)
                # mask[missing_col_id[i]] = False
                print ("fill" + name + "with mean")
                # offset += 1
            else:
                model_list.append(model_dict[name])
                new_missing_col_id.append(missing_col_id[i])

        # now, impute the left missing columns using the model from model_list (ignore the extra columns)
        to_impute_data = data #just change a name..
        missing_col_data = to_impute_data[:, new_missing_col_id]
        epoch = iterations
        counter = 0
        # mean init all missing-value columns
        init_imputation = ["mean"] * len(new_missing_col_id)
        next_data = mvp.imputeData(to_impute_data, new_missing_col_id, init_imputation, self._verbose)

        while (counter < epoch):
            for i in range(len(new_missing_col_id)):
                target_col = new_missing_col_id[i]
                next_data[:, target_col] = missing_col_data[:,i] #recover the column that to be imputed

                next_data = mvp.transform(next_data, target_col, model_list[i], self._verbose)

            counter += 1

        # put back to data
        # data[:, mask] = next_data
        return next_data

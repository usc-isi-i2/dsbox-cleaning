import numpy as np
import pandas as pd
from . import missing_value_pred as mvp
import d3m.metadata.base as mbase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import stopit
import math
import typing

from d3m import container
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import UniformBool
import common_primitives.utils as utils

import typing

from . import config

Input = container.DataFrame
Output = container.DataFrame

# store the regression models for each missing-value column in training data


class IR_Params(params.Params):
    fitted : typing.Union[typing.Any, None]
    verbose : typing.Union[typing.Any, None]
    iterations_done : typing.Union[typing.Any, None]
    has_finished : typing.Union[typing.Any, None]
    best_imputation : typing.Union[typing.Any, None]

class IterativeRegressionHyperparameter(hyperparams.Hyperparams):
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

class IterativeRegressionImputation(UnsupervisedLearnerPrimitiveBase[Input, Output, IR_Params, IterativeRegressionHyperparameter]):
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
        # Required
        "id": "f70b2324-1102-35f7-aaf6-7cd8e860acc4",
        "version": config.VERSION,
        "name": "DSBox Iterative Regression Imputer",
        "description": "Impute missing values using iterative regression",
        "python_path": "d3m.primitives.dsbox.IterativeRegressionImputation",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["IMPUTATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [config.REPOSITORY]
        },
        # Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        # Optional
        "keywords": ["preprocessing", "imputation"],
        "installation": [config.INSTALLATION],
        "location_uris": [],
        "precondition": [hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES],
        "effects": [hyperparams.base.PrimitiveEffect.NO_MISSING_VALUES],
        "hyperparms_to_tune": []
    })

    def __init__(self, *, hyperparams: IterativeRegressionHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)

        # All primitives must define these attributes
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._best_imputation: typing.Dict = {}  # in params.regression_models
        self._train_x: Input = None
        self._is_fitted = True
        self._has_finished = True
        self._iterations_done = True
        self._verbose = hyperparams['verbose'] if hyperparams else False

    def set_params(self, *, params: IR_Params) -> None:
        self._is_fitted = params['fitted']
        self._verbose = params['verbose']
        self._iterations_done = params['iterations_done']
        self._has_finished = params['has_finished']
        self._best_imputation = params['best_imputation']

    def get_params(self) -> IR_Params:
        return IR_Params(
            fitted = self._is_fitted,
            verbose = self._verbose,
            iterations_done = self._iterations_done,
            has_finished = self._has_finished,
            best_imputation = self._best_imputation
            )

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
            timeout = 2**31 - 1
        if (iterations is None):
            self._iterations_done = True
            iterations = 30

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            data = self._train_x.copy()

            # start fitting
            if self._verbose:
                print("=========> iteratively regress method:")
            data_clean, self._best_imputation = self.__iterativeRegress(data, iterations)

        # self._train_x, self._best_imputation = self.__iterativeRegress(data, iterations)
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

        # inputs = inputs.convert_objects(convert_numeric=True)
        attribute = utils.list_columns_with_semantic_types(
            inputs.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])
        numeric = utils.list_columns_with_semantic_types(
            inputs.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        numeric = [x for x in numeric if x in attribute]

        # keys = data.keys()
        # missing_col_id = []
        inputs = inputs.iloc[:, numeric].apply(
            lambda col: pd.to_numeric(col, errors='coerce'))
        # data = mvp.df2np(numeric_data, missing_col_id, self._verbose)

        for i in numeric:
            old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
            old_metadata["structural_type"] = inputs.iloc[:, i].values.dtype.type
            inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        # Impute numerical attributes only

        if (not self._is_fitted):
            # todo: specify a NotFittedError, like in sklearn
            raise ValueError("Calling produce before fitting.")

        if (pd.isnull(inputs).sum().sum() == 0):    # no missing value exists
            if self._verbose:
                print("Warning: no missing value in test dataset")
            self._has_finished = True
            return CallResult(inputs, self._has_finished, self._iterations_done)

        if (timeout is None):
            timeout = 2**31 - 1
        if (iterations is None):
            self._iterations_done = True
            iterations = 30  # only works for iteratively_regre method

        data = inputs.copy()
        # record keys:
        keys = data.keys()
        index = data.index

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose:
                print("=========> iteratively regress method:")
            data_clean = self.__regressImpute(data, self._best_imputation, iterations)
        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._has_finished = True
            value = pd.DataFrame(data_clean, index, keys)
            value = container.DataFrame(value)
            value.metadata = data.metadata
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            print("Timed Out...")
            self._is_fitted = False
            self._has_finished = False
            self._iterations_done = False
        return CallResult(value, self._has_finished, self._iterations_done)


    @classmethod
    def _get_columns_to_fit(cls, inputs: Input, hyperparams: IterativeRegressionHyperparameter):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce


    @classmethod
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int, hyperparams: IterativeRegressionHyperparameter) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False


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
        numeric_data = data.iloc[:, numeric].apply(
            lambda col: pd.to_numeric(col, errors='coerce'))
        data = mvp.df2np(numeric_data, missing_col_id, self._verbose)

        # Impute numerical attributes only
        missing_col_id = [x for x in missing_col_id if x in numeric]
        missing_col_data = data[:, missing_col_id]

        # If all values in a column are missing, set that column to zero
        all_missing = np.sum(np.isnan(missing_col_data), axis=0) == missing_col_data.shape[0]
        for col, col_missing in enumerate(all_missing):
            if col_missing:
                missing_col_data[:, col] = 0

        imputed_data = np.zeros([data.shape[0], len(missing_col_id)])
        imputed_data_lastIter = missing_col_data
        # coeff_matrix = np.zeros([len(missing_col_id), data.shape[1]-1]) #coefficient vector for each missing value column
        model_list = [None] * len(missing_col_id)     # store the regression model
        epoch = iterations
        counter = 0
        # mean init all missing-value columns
        init_imputation = ["mean"] * len(missing_col_id)
        next_data = mvp.imputeData(data, missing_col_id, init_imputation, self._verbose)

        while (counter < epoch):
            for i in range(len(missing_col_id)):
                target_col = missing_col_id[i]
                next_data[:, target_col] = missing_col_data[:, i]  # recover the column that to be imputed

                data_clean, model_list[i] = mvp.bayeImpute(next_data, target_col, self._verbose)
                next_data[:, target_col] = data_clean[:, target_col]    # update bayesian imputed column
                imputed_data[:, i] = data_clean[:, target_col]    # add the imputed data

                # if (is_eval):
                #     self.__evaluation(data_clean, label)

            # if (counter > 0):
            #     distance = np.square(imputed_data - imputed_data_lastIter).sum()
            #     if self._verbose: print("changed distance: {}".format(distance))
            imputed_data_lastIter = np.copy(imputed_data)
            counter += 1
        data[:, missing_col_id] = imputed_data_lastIter
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

        model_list = []  # the model list
        new_missing_col_id = []  # the columns that have correspoding model
        # mask = np.ones((data.shape[1]), dtype=bool)   # false means: this column cannot be bring into impute
        # offset = 0  # offset from missing_col_id to new_missing_col_id

        for i in range(len(missing_col_id)):
            name = col_names[missing_col_id[i]]
            # if there is a column that not appears in trained model, impute it as "mean"
            if (name not in model_dict.keys()):
                data = mvp.imputeData(data, [missing_col_id[i]], ["mean"], self._verbose)
                # mask[missing_col_id[i]] = False
                print("fill" + name + "with mean")
                # offset += 1
            else:
                model_list.append(model_dict[name])
                new_missing_col_id.append(missing_col_id[i])

        # now, impute the left missing columns using the model from model_list (ignore the extra columns)
        to_impute_data = data  # just change a name..
        missing_col_data = to_impute_data[:, new_missing_col_id]
        epoch = iterations
        counter = 0
        # mean init all missing-value columns
        init_imputation = ["mean"] * len(new_missing_col_id)
        next_data = mvp.imputeData(to_impute_data, new_missing_col_id, init_imputation, self._verbose)

        while (counter < epoch):
            for i in range(len(new_missing_col_id)):
                target_col = new_missing_col_id[i]
                next_data[:, target_col] = missing_col_data[:, i]  # recover the column that to be imputed

                next_data = mvp.transform(next_data, target_col, model_list[i], self._verbose)

            counter += 1

        # put back to data
        # data[:, mask] = next_data
        return next_data

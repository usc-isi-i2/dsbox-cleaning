import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from . import missing_value_pred as mvp

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import stopit
import math
import typing

from d3m import container
from d3m.metadata import hyperparams, params
from d3m.metadata.hyperparams import UniformBool
import d3m.metadata.base as mbase
import common_primitives.utils as utils

from . import config

Input = container.DataFrame
Output = container.DataFrame

# store the best imputation strategy for each missing-value column in training data


class Params(params.Params):
    greedy_strategy: typing.Dict


class GreedyHyperparameter(hyperparams.Hyperparams):
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

class GreedyImputation(SupervisedLearnerPrimitiveBase[Input, Output, Params, GreedyHyperparameter]):
    """
    Impute the missing value by greedy search of the combinations of standalone simple imputation method.

    Parameters:
    ----------
    verbose: bool
        Control the verbosity

    Attributes:
    ----------
    imputation_strategies: list of string,
        each is a standalone simple imputation method

    best_imputation: dict. key: column name; value: trained imputation method (parameters)
            which is one of the imputation_strategies

    model: a sklearn machine learning class
        The machine learning model that will be used to evaluate the imputation strategies

    """

    metadata = hyperparams.base.PrimitiveMetadata({
        # Required
        "id": "ebebb1fa-a20c-38b9-9f22-bc92bc548c19",
        "version": config.VERSION,
        "name": "DSBox Greedy Imputer",
        "description": "Impute missing values using greedy search, supervised learining",

        "python_path": "d3m.primitives.data_preprocessing.GreedyImputation.DSBOX",
        "primitive_family": "DATA_PREPROCESSING",
        "algorithm_types": ["IMPUTATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        # Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        # Optional
        "keywords": ["preprocessing", "imputation", "greedy"],
        "installation": [config.INSTALLATION],
        "location_uris": [],
        "precondition": [hyperparams.base.PrimitivePrecondition.NO_CATEGORICAL_VALUES],
        # "effects": [hyperparams.base.PrimitiveEffects.NO_MISSING_VALUES],
        "hyperparms_to_tune": []
    })

    def __init__(self, *, hyperparams: GreedyHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        # All primitives must define these attributes
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._imputation_strategies = ["mean", "max", "min", "zero"]
        self._best_imputation: typing.Dict = {}  # in params.regression_models
        self._train_x: Input = None
        self._train_y: Input = None
        self._is_fitted = True
        self._has_finished = True
        self._iterations_done = True
        self._verbose = hyperparams['verbose'] if hyperparams else False

    def set_params(self, *, params: Params) -> None:
        self._is_fitted = "greedy_strategy" in params
        self._has_finished = self._is_fitted
        self._best_imputation = params['greedy_strategy']

    def get_params(self) -> Params:
        if self._is_fitted:
            return Params(greedy_strategy=self._best_imputation)
        else:
            return Params(greedy_strategy=dict())

    def set_training_data(self, *, inputs: Input, outputs: Output) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Input
            The inputs.
        outputs : Output
            The outputs.
        """
        self._train_x = inputs
        self._train_y = outputs
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

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            if isinstance(self._train_x, pd.DataFrame):
                data = self._train_x.copy()
                label = self._train_y.copy()
            else:
                data = self._train_x[0].copy()
                label = self._train_y[0].copy()

            # start fitting...
            # 1. to figure out what kind of problem it is and assign model and scorer
            # now only support "classification" or "regresion" problem
            self._set_model_scorer()
            # 2. using the model and scorer to do greedy search
            if self._verbose:
                print("=========> Greedy searched imputation:")
            self._best_imputation = self.__imputationGreedySearch(data, label)

        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._has_finished = True
            self._iterations_done = True
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            print("Timed Out...")
            self._is_fitted = False
            self._has_finished = False
            self._iterations_done = False
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

        if (timeout is None):
            timeout = 2**31 - 1

        data = inputs.copy()

        # record keys:
        keys = data.keys()
        index = data.index

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if self._verbose:
                print("=========> impute using result from greedy search:")
            data_clean = self.__simpleImpute(data, self._best_imputation)

        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._is_fitted = True
            self._has_finished = True
            self._iterations_done = True
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
    def _get_columns_to_fit(cls, inputs: Input, hyperparams: GreedyHyperparameter):
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
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int, hyperparams: GreedyHyperparameter) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    #============================================ fit phase functinos ============================================
    def _set_model_scorer(self, model=None, scorer=None):
        """
        figure out what model and scorer should be used for given dataset (label)
        also possible to mannually set
        """
        if (model is not None) and (scorer is not None):
            self.model = model
            self.scorer = scorer
            return

        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.svm import SVR  # type: ignore
        from sklearn.metrics import f1_score, make_scorer, r2_score  # type: ignore

        # set default scorer
        is_classification = self.__isCat_95in10(self._train_y)
        if is_classification == True:
            self.model = LogisticRegression()
            self.scorer = make_scorer(f1_score, average="macro")
        else:
            self.model = SVR()
            self.scorer = make_scorer(r2_score, greater_is_better=False)  # score will be * -1, if greater_is_better is set to False

    def __imputationGreedySearch(self, data, label):
        """
        running greedy search for imputation combinations
        """

        # indices for numeric attribute columns only
        attribute = utils.list_columns_with_semantic_types(
            data.metadata, ['https://metadata.datadrivendiscovery.org/types/Attribute'])
        numeric = utils.list_columns_with_semantic_types(
            data.metadata, ['http://schema.org/Integer', 'http://schema.org/Float'])
        numeric = [x for x in numeric if x in attribute]

        col_names = data.keys()
        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data = mvp.df2np(data, missing_col_id, self._verbose)

        # Imput numerical attribute columns only. Should consider imputing category attribute
        missing_col_id = [x for x in missing_col_id if x in numeric]

        label = label.values

        # init for the permutation
        permutations = [0] * len(missing_col_id)   # length equal with the missing_col_id; value represents the id for imputation_strategies
        pos = len(permutations) - 1
        min_score = float("inf")
        max_score = -float("inf")
        max_strategy_id = 0
        best_combo = [0] * len(missing_col_id)  # init for best combo

        # greedy search for the best permutation
        iteration = 1
        while (iteration > 0):
            for i in range(len(permutations)):
                max_strategy_id = permutations[i]

                for strategy in range(len(self._imputation_strategies)):
                    permutations[i] = strategy
                    imputation_list = [self._imputation_strategies[x] for x in permutations]

                    data_clean = mvp.imputeData(data, missing_col_id, imputation_list, self._verbose)
                    if self._verbose:
                        print("for the missing value imputation combination: {} ".format(permutations))
                    score = self.__evaluation(data_clean, label)
                    if (score > max_score):
                        max_score = score
                        max_strategy_id = strategy
                        best_combo = permutations
                    min_score = min(score, min_score)

                permutations[i] = max_strategy_id

            iteration -= 1

        if self._verbose:
            print("max score is {}, min score is {}\n".format(max_score, min_score))
            print("and the best score is given by the imputation combination: ")

        best_imputation = {}    # key: col_name; value: imputation strategy
        for i in range(len(best_combo)):
            best_imputation[col_names[missing_col_id[i]]] = self._imputation_strategies[best_combo[i]]
            if self._verbose:
                print(self._imputation_strategies[best_combo[i]] + " for the column {}".format(col_names[missing_col_id[i]]))

        return best_imputation

    #============================================ helper  functions ============================================
    def __isCat_95in10(self, label):
        """
        copied from dsbox.datapreprocessing.cleaner.encoder:
        hardcoded rule for identifying (integer/string) categorical column
        """
        col = label[label.keys()[0]]    # assume only one label
        return col.value_counts().head(10).sum() / float(col.count()) > .95

    def __simpleImpute(self, data, strategies_dict, verbose=False):
        """
        impute the data using given strategies
        Parameters:
        ----------
        data: pandas dataframe
        strategies_dict: dict. maps: col_name -> imputation_method
            imputation strategies combination
        """

        col_names = data.keys()
        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data = mvp.df2np(data, missing_col_id, self._verbose)  # no need for label

        strategies = []  # list of strategies, exactly match with missing_col_id
        # extra missing-value columns occurs, using default "mean";
        # some missing-value columns not occurs, ignore them
        for i in range(len(missing_col_id)):
            name = col_names[missing_col_id[i]]
            if (name not in strategies_dict.keys()):
                strategies.append("mean")
            else:
                strategies.append(strategies_dict[name])

        # 2. impute data
        data_clean = mvp.imputeData(data, missing_col_id, strategies, verbose)

        return data_clean

    def __evaluation(self, data_clean, label):
        """
        using defined model and scorer to evaluation the cleaned dataset

        Parameters:
        ----------
        data_clean: the clean dataset, missing values imputed already
        label: the label for data_clean
        """
        from sklearn.model_selection import train_test_split  # type: ignore
        try:
            X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=0, stratify=label)
        except Exception:
            if self._verbose:
                print("cannot stratified sample, try random sample: ")
            X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=42)

        # remove the nan rows

        mask_train = np.isnan(X_train).any(axis=1)  # nan rows index
        mask_test = np.isnan(X_test).any(axis=1)
        num_removed_test = sum(mask_test)
        X_train = X_train[~mask_train]
        y_train = y_train[~mask_train]
        X_test = X_test[~mask_test]
        y_test = y_test[~mask_test]

        model = self.model.fit(X_train, y_train.ravel())
        score = self.scorer(model, X_test, y_test)  # refer to sklearn scorer: score will be * -1 with the real score value
        if self._verbose:
            print("score is: {}".format(score))

        if self._verbose:
            print("===========>> max score is: {}".format(score))
        if (num_removed_test > 0):
            print("BUT !!!!!!!!there are {} data (total test size: {})that cannot be predicted!!!!!!\n".format(num_removed_test, mask_test.shape[0]))
        return score

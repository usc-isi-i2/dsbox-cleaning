import numpy as np
import pandas as pd
from . import missing_value_pred as mvp

from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallMetadata
from typing import NamedTuple, Sequence
import stopit
import math

Input = pd.DataFrame
Output = pd.DataFrame

Params = NamedTuple("Params", [
    ('greedy_strategy', dict)]
    )


class GreedyImputation(SupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    Impute the missing value by greedy search of the combinations of standalone simple imputation method.

    Parameters:
    ----------
    verbose: Integer
        Control the verbosity

    Attributes:
    ----------
    imputation_strategies: list of string,
        each is a standalone simple imputation method

    best_imputation: dict. key: column name; value: trained imputation method (parameters)
            which is one of the imputation_strategies

    model: a sklearn machine learning class
        The machine learning model that will be used to evaluate the imputation strategies

    scorer: a sklearn metrics class
        The metrics that will be used

    """

    def __init__(self, verbose=0) -> None:
        self.best_imputation = None
        self.imputation_strategies = ["mean", "max", "min", "zero"]
        self.train_x = None
        self.train_y = None
        self.is_fitted = True
        self._has_finished = True
        self._iterations_done = True
        self.verbose = verbose


    def set_params(self, *, params: Params) -> None:
        self.is_fitted = len(params.greedy_strategy) > 0
        self._has_finished = self.is_fitted
        self.best_imputation = params.greedy_strategy

    def get_params(self) -> Params:
        if self.is_fitted:
            return Params(greedy_strategy=self.best_imputation)
        else:
            return Params(greedy_strategy=dict())


    def get_call_metadata(self) -> CallMetadata:
            return CallMetadata(has_finished=self._has_finished, iterations_done=self._iterations_done)


    def set_training_data(self, *, inputs: Sequence[Input], outputs: Sequence[Output]) -> None:
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs.
        outputs : Sequence[Output]
            The outputs.
        """
        self.train_x = inputs
        self.train_y = outputs
        self.is_fitted = False



    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
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
        if self.is_fitted:
            return True

        if (timeout is None):
            timeout = math.inf

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            if isinstance(self.train_x, pd.DataFrame):
                data = self.train_x.copy()
                label = self.train_y.copy()
            else:
                data = self.train_x[0].copy()
                label = self.train_y[0].copy()

            # start fitting...
            # 1. to figure out what kind of problem it is and assign model and scorer
            # now only support "classification" or "regresion" problem
            self.set_model_scorer()
            # 2. using the model and scorer to do greedy search
            if (self.verbose > 0): print("=========> Greedy searched imputation:")
            self.best_imputation = self.__imputationGreedySearch(data, label)


        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self.is_fitted = True
            self._has_finished = True
            self._iterations_done = True
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            print ("Timed Out...")
            self.is_fitted = False
            self._has_finished = False
            self._iterations_done = False
            return


    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
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

        if (not self.is_fitted):
            # todo: specify a NotFittedError, like in sklearn
            raise ValueError("Calling produce before fitting.")

        if (timeout is None):
            timeout = math.inf

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        else:
            data = inputs[0].copy()

        # record keys:
        keys = data.keys()

        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

            # start completing data...
            if (self.verbose>0): print("=========> impute using result from greedy search:")
            data_clean = self.__simpleImpute(data, self.best_imputation)


        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self.is_fitted = True
            self._has_finished = True
            self._iterations_done = True
            return pd.DataFrame(data=data_clean, columns=keys)
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            print("Timed Out...")
            self.is_fitted = False
            self._has_finished = False
            self._iterations_done = False
            return None



    #============================================ fit phase functinos ============================================
    def set_model_scorer(self, model=None, scorer=None):
        """
        figure out what model and scorer should be used for given dataset (label)
        also possible to mannually set
        """
        if (model is not None) and (scorer is not None):
            self.model = model
            self.scorer = scorer
            return

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVR
        from sklearn.metrics import f1_score, make_scorer, r2_score


        is_classification = self.__isCat_95in10(self.train_y)
        if is_classification == True:
            self.model = LogisticRegression()
            self.scorer = make_scorer(f1_score, average="macro")
        else:
            self.model = SVR()
            self.scorer = make_scorer(r2_score, greater_is_better=False) # score will be * -1, if greater_is_better is set to False




    def __imputationGreedySearch(self, data, label):
        """
        running greedy search for imputation combinations
        """

        col_names = data.keys()
        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data = mvp.df2np(data, missing_col_id, self.verbose)
        label = label.values

        # init for the permutation
        permutations = [0] * len(missing_col_id)   # length equal with the missing_col_id; value represents the id for imputation_strategies
        pos = len(permutations) - 1
        min_score = float("inf")
        max_score = -float("inf")
        max_strategy_id = 0
        best_combo = [0] * len(missing_col_id)  #init for best combo

        # greedy search for the best permutation
        iteration = 1
        while (iteration > 0):
            for i in range(len(permutations)):
                max_strategy_id = permutations[i]

                for strategy in range(len(self.imputation_strategies)):
                    permutations[i] = strategy
                    imputation_list = [self.imputation_strategies[x] for x in permutations]

                    data_clean = mvp.imputeData(data, missing_col_id, imputation_list, self.verbose)
                    if (self.verbose>0): print("for the missing value imputation combination: {} ".format(permutations))
                    score = self.__evaluation(data_clean, label)
                    if (score > max_score):
                        max_score = score
                        max_strategy_id = strategy
                        best_combo = permutations
                    min_score = min(score, min_score)

                permutations[i] = max_strategy_id

            iteration -= 1


        if (self.verbose>0):
            print("max score is {}, min score is {}\n".format(max_score, min_score))
            print("and the best score is given by the imputation combination: ")

        best_imputation = {}    # key: col_name; value: imputation strategy
        for i in range(len(best_combo)):
            best_imputation[col_names[missing_col_id[i]]] = self.imputation_strategies[best_combo[i]]
            if (self.verbose>0):
                print(self.imputation_strategies[best_combo[i]] + " for the column {}".format(col_names[missing_col_id[i]]))


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
        data = mvp.df2np(data, missing_col_id, self.verbose) # no need for label

        strategies = [] # list of strategies, exactly match with missing_col_id
        # extra missing-value columns occurs, using default "mean";
        # some missing-value columns not occurs, ignore them
        for i in range(len(missing_col_id)):
            name = col_names[missing_col_id[i]]
            if (name not in strategies_dict.keys()):
                strategies.append("mean")
            else:
                strategies.append(strategies_dict[name])

        print(strategies)
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
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=0, stratify=label)
        except:
            if (self.verbose>0): print("cannot stratified sample, try random sample: ")
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
        if (self.verbose>0): print("score is: {}".format(score))

        if (self.verbose>0): print("===========>> max score is: {}".format(score))
        if (num_removed_test > 0):
            print("BUT !!!!!!!!there are {} data (total test size: {})that cannot be predicted!!!!!!\n".format(num_removed_test, mask_test.shape[0]))
        return score

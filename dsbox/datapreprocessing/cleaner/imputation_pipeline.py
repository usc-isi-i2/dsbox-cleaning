import numpy as np
import pandas as pd

import missing_value_pred as mvp
import helper_func as hf


class Imputation(object):
    """search for best Imputation combination for a given dataset 

    Parameters:
    ----------
    strategies: list of string
        The Imputation strategies that are considered to be searched.

    verbose: Integer
        Control the verbosity

    eva_model: string
        The machine learning model that will be used to evaluate the imputation strategies
            support: 'linearSVM', 'decisionTree', 'logisticRegreesion', 


    Attributes:
    ----------
    best_imputation: list of string
        After 'fit' method, the best imputation combination will be given.
    """

    def __init__(self, model, scorer, greater_is_better=True, verbose=0, strategies=["mean", "max", "min", "zero"]):
        self.allowed_strategies = ["mean", "max", "min", "zero"] 
        self.verbose = verbose
        self.imputation_strategies = strategies
        self.model = model
        self.scorer = scorer


    def fit(self, data, label):
        """
        imputation combination evaluations
        now, only for classification problem (because I convert the label to integer)

        Parameters:
        ----------
        data: pandas dataframe
        label: pandas dataframe, assume second column is the label targets
        """
        
        # 1. convert categorical to indicator
        label_col_name = label.keys()[1]    # assume the second column is the label targets
        
        for col_name in data:
            if(mvp.isCategorical(data[col_name]) != None):
                data = hf.cate2ind(data, col_name)
        # convert the label also, but not to indicator, convert to integer
        data[label_col_name] = label[label_col_name] 

        if (data[label_col_name].dtypes != int and data[label_col_name].dtypes != float):
            cate_map = mvp.cate2int(data[label_col_name].unique())
            if self.verbose:
                print "convert label to integer: {}".format(cate_map)
            data[label_col_name] = data[label_col_name].replace(cate_map)

        # 2. start evaluation
        print "=========> Baseline:"
        self.__baseline(data, label_col_name)
        # print "=========> Greedy searched imputation:"
        # self.__imputationGreedy(data, label_col_name)
        print "=========> other imputation method:"
        self.__otherImpute(data, label_col_name)

        return self.best_imputation

    def impute(self, data, strategies, verbose):
        """
        impute the data using given strategies
        Parameters:
        ----------
        data: pandas dataframe
        strategies: list of string
            imputation strategies combination
        """
        for each in strategies:
            if each not in self.allowed_strategies:
                raise ValueError("Can only use these strategies: {0} "
                             " got strategy '{1}' ".format(self.allowed_strategies,
                                                        each))

        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data, label = self.__df2np(data, "", missing_col_id) # no need for label
        if (len(missing_col_id) != len(strategies)):
            raise ValueError("Expected {0} number of permutations, "
                             " got '{1}' ".format(len(missing_col_id),
                                                        len(strategies)))
        # 2. impute data
        data_clean = mvp.imputeData(data, missing_col_id, strategies, verbose)

        return data_clean

    def __otherImpute(self, data, label_col_name):
        from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
        missing_col_id = []
        data, label = self.__df2np(data, label_col_name, missing_col_id)
        # data_clean = KNN(k=5).complete(data)
        data_clean = NuclearNormMinimization().complete(data)
        self.__evaluation(data_clean, label)


    def __baseline(self, data, label_col_name):
        """
        running baseline
        """
        data_dropCol = data.dropna(axis=1, how="any") #drop the col with nan

        label = data[label_col_name].values
        data = data.drop(label_col_name,axis=1).values  #convert to np array
        label_dropCol = data_dropCol[label_col_name].values
        data_dropCol = data_dropCol.drop(label_col_name,axis=1).values

        #========================STEP 2: pred==============
        print "==============result for baseline: drop rows============"
        self.__evaluation(data, label)
        print "==============result for baseline: drop columns============"

        self.__evaluation(data_dropCol, label_dropCol)
        print "========================================================"


    def __df2np(self, data, label_col_name, missing_col_id):
        """
        helper function: convert dataframe to np array
        """
        counter = 0    
        
        # 1. get the id for missing value column
        for col_name in data:
            num = sum(pd.isnull(data[col_name]))
            if (num > 0):
                missing_col_id.append(counter)
            counter += 1

        # 2. convert the dataframe to np array
        label = None
        col_names = data.keys()
        if (len(label_col_name)>0):
            label = data[label_col_name].values
        data = data.drop(label_col_name,axis=1).values  #convert to np array

        return data, label

    def __imputationGreedy(self, data, label_col_name):
        """
        running greedy search for imputation combinations
        """        
        for each in self.imputation_strategies:
            if each not in self.allowed_strategies:
                raise ValueError("Can only use these strategies: {0} "
                             " got strategy '{1}' ".format(self.allowed_strategies,
                                                        each))
        
        col_names = data.keys()
        # 1. convert to np array and get missing value column id
        missing_col_id = []
        data, label = self.__df2np(data, label_col_name, missing_col_id)
        
        # init for the permutation
        permutations = [0] * len(missing_col_id)   # length equal with the missing_col_id; value represents the id for imputation_strategies
        pos = len(permutations) - 1
        min_score = float("inf")
        max_score = -float("inf")
        max_strategy_id = 0  
        best_combo = [0] * len(missing_col_id)  #init for best combo

        for i in range(len(permutations)):
            for strategy in range(len(self.imputation_strategies)):
                permutations[i] = strategy
                imputation_list = [self.imputation_strategies[x] for x in permutations]

                data_clean = mvp.imputeData(data, missing_col_id, imputation_list, self.verbose)
                print "for the missing value imputation combination: {} ".format(permutations)
                score = self.__evaluation(data_clean, label)
                if (score > max_score):
                    max_score = score
                    max_strategy_id = strategy
                    best_combo = permutations
                min_score = min(score, min_score)

            permutations[i] = max_strategy_id
            max_strategy_id = 0

        print "max score is {}, min score is {}\n".format(max_score, min_score)
        print "and the best score is given by the imputation combination: "
        for i in range(len(best_combo)):
            print self.imputation_strategies[best_combo[i]] + " for the column {}".format(col_names[missing_col_id[i]])

        self.best_imputation = [self.imputation_strategies[i] for i in best_combo]

    def __evaluation(self, data_clean, label):
        """
        INPUT
        data_clean: the clean dataset, missing values imputed already
        label: the label for data_clean
        """ 
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=0, stratify=label)
        # X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=0)
        # remove the nan rows

        mask_train = np.isnan(X_train).any(axis=1)  # nan rows index
        mask_test = np.isnan(X_test).any(axis=1)
        num_removed_test = sum(mask_test)
        X_train = X_train[~mask_train]
        y_train = y_train[~mask_train]
        X_test = X_test[~mask_test]
        y_test = y_test[~mask_test]

        model = self.model.fit(X_train, y_train)
        score = self.scorer(model, X_test, y_test)  # refer to sklearn scorer: score will be * -1 with the real score value
        print "score is: {}".format(score)

        print "===========>> max score is: {}".format(score)
        if (num_removed_test > 0):
            print "BUT !!!!!!!!there are {} data (total test size: {})that cannot be predicted!!!!!!\n".format(num_removed_test, y_test.shape[0])
        return score

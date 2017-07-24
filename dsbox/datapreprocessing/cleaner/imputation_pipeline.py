import numpy as np
import pandas as pd

import missing_value_pred as mvp


class Imputation(object):
    """
    Integrated imputation methods moduel.

    Parameters:
    ----------
    model: a function
        The machine learning model that will be used to evaluate the imputation strategies

    scorer: a function
        The metrics that will be used

    greater_is_better: boolean
        Indicate whether higher or lower the score is better. Default is True. Usually, for regression problem
        this should be set to False.

    verbose: Integer
        Control the verbosity

    Attributes:
    ----------
    best_imputation: trained imputation method (parameters)
    """

    def __init__(self, model, scorer, greater_is_better=True, verbose=0):
        self.imputation_strategies = ["mean", "max", "min", "zero"] 
        self.verbose = verbose
        self.model = model
        self.scorer = scorer
        self.is_fitted = False


    # for now, make it internally
    def __analysis(self, data, label):
        """
        TODO
        provide some analysis for the missing pattern:
        is missing/not related to other column ?
        """
        data = data.copy()  
        label_col_name = "target_label" #   name for label, assume no duplicate exists in data
        data[label_col_name] = label

         # start evaluation
        print "=========> Baseline:"
        self.__baseline(data, label_col_name)


    def fit(self, data, label=pd.Series(), strategy="greedy"):
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
        data = data.copy()
        if (not label.empty):
            label_col_name = "target_label" #   name for label, assume no duplicate exists in data
            data[label_col_name] = label

        self.best_imputation = None # store the info of trained imputation method

        # start fitting
        if (strategy=="greedy"):
            if (label.empty):
                raise ValueError("label is nessary for greedy search")

            print "=========> Greedy searched imputation:"
            self.best_imputation = self.__imputationGreedySearch(data, label_col_name)

        elif (strategy=="iteratively_regre"):
            print "=========> iteratively regress method:"
            # no operation here because this method not needs to be trained

        elif(strategy=="other"):
            print "=========> other method:"
            # no operation here because this method not needs to be trained
        
        else:
            raise ValueError("no such strategy: {}".format(strategy))

        self.strategy = strategy
        self.is_fitted = True

    def transform(self, data, label=pd.Series()):
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
        data = data.copy()
        # record keys:
        keys = data.keys()
        if (not self.is_fitted):
            # todo: specify a NotFittedError, like in sklearn
            raise ValueError("imputer is not fitted yet")

        label_col_name = ""
        if (not label.empty):
            label_col_name = "target_label" #   name for label, assume no duplicate exists in data
            data[label_col_name] = label

        # start complete data
        if (self.strategy=="greedy"):
            print "=========> impute using result from greedy search:"
            data_clean = self.__simpleImpute(data, self.best_imputation)

        elif (self.strategy=="iteratively_regre"):
            print "=========> iteratively regress method:"
            data_clean, placeholder = self.__iterativeRegress(data, label_col_name)

        elif(self.strategy=="other"):
            print "=========> other method:"
            data_clean = self.__otherImpute(data)

        return pd.DataFrame(data=data_clean, columns=keys)


    #============================================ fit phase functinos ============================================
    def __iterativeRegress(self, data, label_col_name=""):
        '''
        init with simple imputation, then apply regression to impute iteratively
        '''
        if (label_col_name==None or len(label_col_name)==0):
            is_eval = False
        else:
            is_eval = True
        
        missing_col_id = []
        data, label = self.__df2np(data, label_col_name, missing_col_id)
        next_data = data
        missing_col_data = data[:, missing_col_id]
        imputed_data = np.zeros([data.shape[0], len(missing_col_id)])
        imputed_data_lastIter = missing_col_data
        # coeff_matrix = np.zeros([len(missing_col_id), data.shape[1]-1]) #coefficient vector for each missing value column
        model_list = [None]*len(missing_col_id)     # store the regression model
        epoch = 30
        counter = 0
        init_imputation = ["mean"] * len(missing_col_id)   # mean init

        while (counter < epoch):
            for i in range(len(missing_col_id)):
                target_col = missing_col_id[i]
                data_imputed = mvp.imputeData(next_data, missing_col_id, init_imputation, self.verbose)
                data_imputed[:, target_col] = missing_col_data[:,i] #revover the column that to be imputed

                data_clean, model_list[i] = mvp.bayeImpute(data_imputed, target_col)
                next_data[:,target_col] = data_clean[:,target_col]    # update bayesian imputed column
                imputed_data[:,i] = data_clean[:,target_col]    # add the imputed data

                if (is_eval): 
                    self.__evaluation(data_clean, label)

            if (counter > 0):
                distance = np.square(imputed_data - imputed_data_lastIter).sum()
                print "changed distance: {}".format(distance)
            imputed_data_lastIter = np.copy(imputed_data)
            counter += 1

        data[:,missing_col_id] = imputed_data_lastIter

        return data, model_list

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

    def __imputationGreedySearch(self, data, label_col_name):
        """
        running greedy search for imputation combinations
        """        
        
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

        # greedy search for the best permutation
        iteration = 1
        while (iteration > 0):
            for i in range(len(permutations)):
                max_strategy_id = permutations[i]

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

            iteration -= 1

        print "max score is {}, min score is {}\n".format(max_score, min_score)
        print "and the best score is given by the imputation combination: "
        for i in range(len(best_combo)):
            print self.imputation_strategies[best_combo[i]] + " for the column {}".format(col_names[missing_col_id[i]])

        best_imputation = [self.imputation_strategies[i] for i in best_combo]
        return best_imputation

    #============================================ transform phase functions ============================================

    def __simpleImpute(self, data, strategies, verbose=False):
        """
        impute the data using given strategies
        Parameters:
        ----------
        data: pandas dataframe
        strategies: list of string
            imputation strategies combination
        """

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

    def __otherImpute(self, data, label_col_name=""):
        from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE
        from sklearn.preprocessing import scale

        if (label_col_name==None or len(label_col_name)==0):
            is_eval = False
        else:
            is_eval = True

        missing_col_id = []
        data, label = self.__df2np(data, label_col_name, missing_col_id)
        # mask = np.isnan(data)
        # imputation_list = ["mean"] * len(missing_col_id)
        # data_mean = mvp.imputeData(data, missing_col_id, imputation_list, self.verbose)
        # data_mean = scale(data_mean)
        # data_mean[mask] = np.nan

        # data_clean = KNN(k=5, normalizer=BiScaler).complete(data)
        data_clean = KNN(k=5).complete(data)
        #data_clean = MICE().complete(data)

        if (is_eval): self.__evaluation(data_clean, label)

        return data_clean


    
    #====================== helper functions ======================

    def __df2np(self, data, label_col_name, missing_col_id=[]):
        """
        helper function: convert dataframe to np array; 
            in the meanwhile, provide the id for missing column
        """
        counter = 0    
        
        # 1. get the id for missing value column
        missing_col_name = []
        for col_name in data:
            num = sum(pd.isnull(data[col_name]))
            if (num > 0):
                missing_col_id.append(counter)
                missing_col_name.append(col_name)
            counter += 1

        print "missing column name: {}".format(missing_col_name)

        # 2. convert the dataframe to np array
        label = None
        col_names = data.keys()
        if (len(label_col_name)>0):
            label = data[label_col_name].values
            data = data.drop(label_col_name,axis=1)
        data = data.values  #convert to np array

        return data, label

    

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
            print "cannot stratified sample, try random sample: "
            X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=42)
        
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
            print "BUT !!!!!!!!there are {} data (total test size: {})that cannot be predicted!!!!!!\n".format(num_removed_test, mask_test.shape[0])
        return score

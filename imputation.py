import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

import missing_value_pred as mvp
import helper_func as hf


def test(data_path, label_path, drop_col):
    """
    now, only for classification problem (because I convert the label to integer)
    Parameter:
    drop_col: the column name list that should be dropped before processing (because is a ID, or empty column, etc.)
    """
    # 1. read data and drop
    data = pd.read_csv(data_path)   
    label = pd.read_csv(label_path)  
    for each in drop_col:
        data = data.drop(each, axis=1)

    # 2. convert categorical to indicator
    label_col_name = label.keys()[1]    # assume the second column is the label targets
    
    for col_name in data:
        if(mvp.isCategorical(data[col_name]) != None):
            data = hf.cate2ind(data, col_name)
    # convert the label also, but not to indicator, convert to integer
    data[label_col_name] = label[label_col_name] 
    if (data[label_col_name].dtypes != int):
        cate_map = mvp.cate2int(data[label_col_name].unique())
        data[label_col_name] = data[label_col_name].replace(cate_map)


    # 3. start evaluation
    print "=========> Baseline:"
    #baseline(data, label_col_name)
    print "=========> Greedy searched imputation:"
    imputationGreedy(data, label_col_name)


def baseline(data, label_col_name):
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
    mvp.pred(data, label)
    print "==============result for baseline: drop columns============"

    mvp.pred(data_dropCol, label_dropCol)
    print "========================================================"


def imputationGreedy(data, label_col_name):
    """
    running greedy search for imputation combinations
    """

    imputation_strategies = ["mean", "max", "min", "zero"]  
    
    # 1. get the id for missing value column
    missing_col_id = []
    counter = 0    
    
    for col_name in data:
        num = sum(pd.isnull(data[col_name]))
        if (num > 0):
            missing_col_id.append(counter)
        counter += 1

    # 2. convert the dataframe to np array
    col_names = data.keys()
    label = data[label_col_name].values
    data = data.drop(label_col_name,axis=1).values  #convert to np array

    # init for the permutation
    permutations = [0] * len(missing_col_id)   # length equal with the missing_col_id; value represents the id for imputation_strategies
    pos = len(permutations) - 1
    min_score = 1
    max_score = 0
    max_strategy_id = 0  
    best_combo = [0] * len(missing_col_id)  #init for best combo

    for i in range(len(permutations)):
        for strategy in range(len(imputation_strategies)):
            permutations[i] = strategy

            data_clean = mvp.imputeData(data, permutations, missing_col_id, imputation_strategies)
            print "for the missing value imputation combination: {} ".format(permutations)
            score = mvp.pred(data_clean, label)
            if (score > max_score):
                max_score = score
                max_strategy_id = strategy
                best_combo = permutations
            min_score = min(score, min_score)

        permutations[i] = max_strategy_id
        max_strategy_id = 0

    # permutations = [3, 0, 1, 0, 0, 1] 
    # for i in range(10):
    #     data_clean = mvp.imputeData(data, permutations, missing_col_id, imputation_strategies)
    #     print "for the missing value imputation combination: {} ".format(permutations)
    #     score = mvp.pred(data_clean, label)
    #     print score

    print "max score is {}, min score is {}\n".format(max_score, min_score)
    print "and the best score is given by the imputation combination: "
    for i in range(len(best_combo)):
        print imputation_strategies[best_combo[i]] + " for the column {}".format(col_names[missing_col_id[i]])

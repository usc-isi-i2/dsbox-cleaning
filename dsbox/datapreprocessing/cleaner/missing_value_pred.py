import pandas as pd
import numpy as np

def popular_value(array):
    """
    array: 1D array
    """
    cate = dict()
    for cell in array:
        if (cell in cate):
            cate[cell] += 1
        else:
            cate[cell] = 1

    max_freq = 0
    popular = None
    for key in cate.keys():
        if (cate[key] > max_freq):
            popular = key

    return popular

def myImputer(data, value="zero", verbose=0):
    """
    INPUT:
    data: numpy array, matrix
    value:    string: "mean", "min", "max", "zero", "gaussian"
    """
    index = np.isnan(data)
    data_imputed = np.copy(data)
    data_drop = data[np.logical_not(index)]   #drop nan from data
    inputed_value = 0
    if (value == "zero"):
        inputed_value = 0
    elif (value == "mean"):
        inputed_value = np.mean(data_drop)
    elif (value == "max"):
        inputed_value = np.max(data_drop)
    elif (value == "min"):
        inputed_value = np.min(data_drop)
    elif (value == "new"):
        inputed_value = 0   # 0 is the value that never happens in our categorical map
    elif (value == "popular"):
        inputed_value = popular_value(data_drop)
    # special type of imputed, just return after imputation
    elif (value == "knn"):
        data_clean = KNN(k=5).complete(data)
        return data_clean
    else:
        raise ValueError("no such impute strategy: {}".format(value))

    data_imputed[index] = inputed_value

    if verbose: print("imputed missing value: {}".format(inputed_value))
    return data_imputed


def imputeData(data, missing_col_id, imputation_strategies, verbose):
    """
    impute the data using permutations array.
    INPUT:
    data: numpy array, matrix
    value:    string: "mean", "min", "max", "zero", "gaussian"
    """
    data_clean = np.copy(data)

    for i in range(len(imputation_strategies)):
        strategy = imputation_strategies[i]
        col_id = missing_col_id[i]

        data_clean[:,col_id] = myImputer(data[:,col_id], strategy)


    return data_clean

def bayeImpute(data, target_col):
    '''
    currently, naive bayes.
    return the imputated data, and model coefficient
    '''

    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    model = BayesianRidge()
    # model = LinearRegression()
    # model = RandomForestRegressor()

    original_data = np.copy(data)

    target = data[:, target_col]
    data = np.delete(data, obj=target_col, axis=1)  #remove the missing-value column
    mv_mask = np.isnan(target)
    print("number of imputated cells: {}".format(sum(np.isnan(original_data[:,target_col]))))

    x_test = data[mv_mask]
    x_train = data[~mv_mask]
    y_train = target[~mv_mask]

    model.fit(x_train, y_train)
    result = model.predict(x_test)
    original_data[mv_mask, target_col] = result #put the imputation result back to original data, following the index

    # print("coefficient: {}".format(model.coef_))
    return original_data, model

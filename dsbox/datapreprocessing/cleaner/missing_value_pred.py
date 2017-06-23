import pandas as pd
import numpy as np

def isCategorical(column):
    """
    INPUT
    column: pandas.dataframe series
    prematrue method: to see if 95% value is in 10 category
    if is true, will fill in the cate map that: category -> int; if not, return None
    if the cell value is nan(missing value), ignore it, leave to the imputation later
    """
    #column = column.dropna()
    total_len = len(column)
    cate = dict()
    for cell in column:
        if (len(cate) > 10):
            return None
        if (cell in cate):
            cate[cell] += 1
        else:
            cate[cell] = 1

    sorted_value = sorted(cate.values(), reverse = True)
    top10_ratio = sum(sorted_value[0:10])/float(total_len)
    if (top10_ratio >= 0.95):
        return cate2int(cate.keys())
    else:
        return None

def cate2int(cate):
    """
    return a map (dict) that maps category code to integer: 1,2,3, ... (save 0 for the new value!)
    INPUT
    cate: is a category list
    """
    category_map = {}
    ind = 1
    for i in cate:
        category_map[i] = ind
        ind += 1
    return category_map

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

def myImputer(data, value="zero"):
    """
    INPUT:
    data: numpy array, 1*D (one column)
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

    data_imputed[index] = inputed_value

    #print "imputed missing value: {}".format(inputed_value)
    return data_imputed


def imputeData(data, permutations, missing_col_id, imputation_strategies):
    """
    impute the data using permutations array.
    """
    data_clean = np.copy(data)

    for i in range(len(permutations)):
        strategy = permutations[i]
        col_id = missing_col_id[i]

        data_clean[:,col_id] = myImputer(data[:,col_id], imputation_strategies[strategy])

    return data_clean


#========================two learning & prediction method==============
def pred(data_clean, label):
    """
    INPUT
    data_clean: the clean dataset, missing values imputed already
    label: the label for data_clean
    """
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn import metrics
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(data_clean, label, test_size=0.4, random_state=0, stratify=label)
    # remove the nan rows
    mask_train = np.isnan(X_train).any(axis=1)  # nan rows index
    mask_test = np.isnan(X_test).any(axis=1)
    num_removed_test = sum(mask_test)
    X_train = X_train[~mask_train]
    y_train = y_train[~mask_train]
    X_test = X_test[~mask_test]
    y_test = y_test[~mask_test]

    # classifers:
    clf_linear = svm.SVC(kernel='linear').fit(X_train, y_train)
    clf_lg = LogisticRegressionCV(scoring="f1_macro").fit(X_train, y_train)

    max_score = 0
    print "for linear SVM: "
    print f1_score(y_test, clf_linear.predict(X_test), average=None)
    score = f1_score(y_test, clf_linear.predict(X_test), average="macro")    #weighted average over all the classes
    max_score = max(score, max_score)
    print  score

    print "for LogisticRegressionCV: "
    print f1_score(y_test, clf_lg.predict(X_test), average=None)
    score = f1_score(y_test, clf_lg.predict(X_test), average="macro")   #weighted average over all the classes
    max_score = max(score, max_score)
    print score 

    from sklearn import tree
    clf_tree = tree.DecisionTreeClassifier().fit(X_train, y_train)
    print "for decision tree: "
    print f1_score(y_test, clf_tree.predict(X_test), average=None)
    score = f1_score(y_test, clf_tree.predict(X_test), average="macro")   #weighted average over all the classes
    max_score = max(score, max_score)
    print score 

    print "===========>> max weighted score is: {}".format(max_score)
    if (num_removed_test > 0):
        print "BUT !!!!!!!!there are {} data (total test size: {})that cannot be predicted!!!!!!\n".format(num_removed_test, y_test.shape[0])
    return max_score

def pred_cv(data_clean, label, eva_metrics):
    """
    prediction with cross validation.
    INPUT
    data_clean: the clean dataset, missing values imputed already
    label: the label for data_clean
    """
    from sklearn import svm
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.linear_model import LinearRegression
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    clf_linear = svm.SVC(kernel='linear')
    clf_lg = LogisticRegressionCV(scoring=eva_metrics)
    clf_tree = tree.DecisionTreeClassifier()
    max_score = 0

    score = cross_val_score(clf_linear, data_clean, label, scoring=eva_metrics).mean()
    print "for SVM linear, score is {}".format(score)
    max_score = max(max_score, score)

    score = cross_val_score(clf_lg, data_clean, label, scoring=eva_metrics).mean()
    print "for LogisticRegression, score is {}".format(score)
    max_score = max(max_score, score)

    score = cross_val_score(clf_tree, data_clean, label, scoring=eva_metrics).mean()
    print "for DecisionTree, score is {}".format(score)
    max_score = max(max_score, score)

    print "===========>> max weighted score is: {}".format(max_score)
    return max_score

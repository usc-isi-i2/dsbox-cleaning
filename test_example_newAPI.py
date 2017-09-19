"""
sample program for classification problem
"""
def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn import tree
import pandas as pd

from dsbox.datapreprocessing.cleaner import Imputation

# STEP 1: get data
data_path = "../../dsbox-data/o_38/encoded/"
data_name = data_path + "trainData_encoded.csv"
label_name = data_path + "trainTargets_encoded.csv" # make sure your label target is in the second column of this file



data = pd.read_csv(data_name)
label = text2int(pd.read_csv(label_name)["Class"])

data.drop("d3mIndex",axis=1)    # drop because id, useless

# STEP 2: define your machine learning model and scorer
clf = LogisticRegression()
scorer = make_scorer(f1_score, average="macro") # score will be * -1, if greater_is_better is set to False

# STEP 3: go to use the Imputer !
imputer = Imputation()
imputer.set_params(model=clf, scorer=scorer, strategy="iteratively_regre", verbose=1)
imputer.set_training_data(data,label)
imputer.fit(timeout = 10)
print (imputer.get_call_metadata())
result = imputer.produce(data, timeout=0.5)
# method: greedy search
# data_test = data.drop("age",axis=1)
# imputer.fit(data_test, label)

# data_clean = imputer.transform(data)
# print imputer.best_imputation

# method: regression
# imputer.fit(data)   # on age column, no missing value
# print (imputer.best_imputation.keys())

# data_clean = imputer.transform(data)    # on age column, has missing value

# data_clean.to_csv("data_clean.csv", index=False)

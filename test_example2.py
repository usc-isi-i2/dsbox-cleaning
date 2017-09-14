"""
sample program for regression problem
"""
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer,mean_squared_error
from sklearn import tree
import pandas as pd

from dsbox.datapreprocessing.cleaner import Imputation, Encoder, text2int

# STEP 1: get data
data_path = "../dsbox-data/r_26/original/data/"
data_name = data_path + "trainData_joint.csv"
label_name = data_path + "trainTargets.csv" # make sure your label target is in the second column of this file

enc = Encoder(text2int=True)
enc.fit(data_name)
data = enc.transform(data_name)
label = text2int(pd.read_csv(label_name)["log_randon"])

data.drop("radonFile_index",axis=1)    # drop because id, useless
data.drop("state",axis=1)    # drop because all same, useless
data.drop("state2",axis=1)    # drop because all same, useless
data.drop("stfips",axis=1)    # drop because all same, useless

# STEP 2: define your machine learning model and scorer
clf = LinearRegression()
scorer = make_scorer(mean_squared_error, greater_is_better=False) # score will be * -1, if greater_is_better is set to False

# STEP 3: go to use the Imputer !
# method: greedy search
imputer = Imputation(model=clf, scorer=scorer, strategy="greedy")
imputer.fit(data, label)
data_clean = imputer.transform(data)
# print imputer.best_imputation

# method: regression
# imputer.fit(data, label)  # in current version, not really learned from this step. only evaluation
# data_clean = imputer.transform(data)

data_clean.to_csv("data_clean.csv", index=False)

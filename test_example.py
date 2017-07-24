"""
sample program for classification problem
"""
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn import tree

from dsbox.datapreprocessing.cleaner import Imputation, encoder

# STEP 1: get data
data_path = "../dsbox-data/o_4550/original/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv" # make sure your label target is in the second column of this file

data = encoder.encode(data_name)
label = encoder.encode(label_name,label="class")["class"]

data.drop("MouseID",axis=1)    # drop because id-like, useless
data.drop("d3mIndex",axis=1)    # drop because id, useless

# STEP 2: define your machine learning model and scorer
clf = LogisticRegression()
scorer = make_scorer(f1_score, average="macro") # score will be * -1, if greater_is_better is set to False

# STEP 3: go to use the Imputer !
imputer = Imputation(model=clf, scorer=scorer)
# method: greedy search
# imputer.fit(data, label, strategy="greedy")
# data_clean = imputer.transform(data)
# print imputer.best_imputation

# method: regression
imputer.fit(data, strategy="iteratively_regre")
data_clean = imputer.transform(data)

data_clean.to_csv("data_clean.csv", index=False)

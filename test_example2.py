from sklearn import svm
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from dsbox.datapreprocessing.cleaner import Imputation
from dsbox.datapreprocessing.cleaner import helper_func

data_path = "../dsbox-data/o_196/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv" # make sure your label target is in the second column of this file
# STEP 1: input the column name that is useless in the dataset, eg. id-like column, name column, etc.
drop_col_name = ["d3mIndex"] 

# STEP 2: define your machine learning model and scorer
clf = RandomForestRegressor()
scorer = make_scorer(mean_squared_error, greater_is_better=False) # score will be * -1, if greater_is_better is set to False

# STEP 3: go to use the Imputer !
imputer = Imputation(model=clf, scorer=scorer, strategies=["mean", "max", "min", "zero"], verbose=1)
data, label = helper_func.dataPrep(data_name, label_name, drop_col_name)
best_imputation = imputer.fit(data, label)
print best_imputation

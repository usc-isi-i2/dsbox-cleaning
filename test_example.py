from dsbox.datapreprocessing.cleaner import Imputation
from dsbox.datapreprocessing.cleaner import helper_func

data_path = "../dsbox-data/o_38/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv"
# input the column name that is useless in the dataset, eg. id-like column
drop_col_name = ["d3mIndex"] 

imputer = Imputation(strategies=["mean", "max", "min", "kk"])
data, label = helper_func.dataPrep(data_name, label_name, drop_col_name)
imputer.fit(data, label)



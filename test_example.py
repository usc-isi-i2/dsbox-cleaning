data_path = "../dsbox-data/o_38/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv"
# input the column name that is useless in the dataset, eg. id-like column
drop_col_name = ["d3mIndex"] 


import imputation_pipeline as IP

data, label = IP.dataPrep(data_name, label_name, drop_col_name)
IP.mainTest(data, label)
from imputation import test

data_path = "../dsbox-data/o_38/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv"
drop_col_name = ["TBG", "d3mIndex"] # the column that are not necessary in data file, like id, or empty column

test(data_name,label_name,drop_col_name)
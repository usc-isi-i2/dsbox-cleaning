## Missing value imputer
This component is for missing value imputation. This module is designed to support:

1. multiple ways to impute data, including our self-defined methods.
2. missing pattern related analysis
3. fit (or train) a method on the data, then apply to other data

Now the functionality is limited to:

* one label problem

### Dependencies
[check here](environment.yml)

if you have conda, simply do the following:

```sh
conda-env create .
source activate mvi
python test_example.py
```

### Usage:
see [test_example.py](test_example.py):

```python
"""
sample program for classification problem
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn import tree

from dsbox.datapreprocessing.cleaner import Imputation, encoder

# STEP 1: get data
data_path = "../dsbox-data/o_38/original/data/"
data_name = data_path + "trainData.csv"
label_name = data_path + "trainTargets.csv" # make sure your label target is in the second column of this file

data = encoder.encode(data_name)
label = encoder.encode(label_name,label="Class")["Class"]

data.drop("d3mIndex",axis=1)    # drop because id, useless

# STEP 2: define your machine learning model and scorer
clf = LogisticRegression()
scorer = make_scorer(f1_score, average="macro") # score will be * -1, if greater_is_better is set to False

# STEP 3: go to use the Imputer !
# method: greedy search
# imputer = Imputation(model=clf, scorer=scorer, strategy="greedy")
# imputer.fit(data, label)
# data_clean = imputer.transform(data)
# print imputer.best_imputation

# method: regression
imputer = Imputation(model=clf, scorer=scorer, strategy="iteratively_regre")
imputer.fit(data)
data_clean = imputer.transform(data, label)

data_clean.to_csv("data_clean.csv", index=False)
```


### TODO:
1. finish verbose func
2. [BUG]may happen the situation: the missing columns in train\_data are not matched with the missing columns in test\_data. 

### methods
1. baseline: drop columns or drop rows
2. (for multiple missing-valu columns) greedy search
3. iteratively regression
4. other ([fancyimpute](https://github.com/hammerlab/fancyimpute))

## One-hot encoder
The encoder takes csv file or pandas DataFrame as input, then one-hot encode columns which are considered categorical. (currently:
take a column as category if:
* its dtype is not float and
* 95% of its data fall in 10 values.
* For the rest values (not top 10) with low frequency, put into one column "_others"

Note, currently: 
* For nonnumeric columns which don't fall into categories, they are converted into integer codes (0,1,2...), just as a temporary expedient.
* For column which has single unique value with some cells missing, the encoder only
  convert the original column into an indicator column "_nan" to tell if missing.
* For column which has two unique values and no cells missing, the encoder only convert
  the original column into binary (0/1) values.

### Usage:
```python
from dsbox.datapreprocessing.cleaner import encoder
# csv file as input: 
result = encoder.encode('yourDataset.csv')

# DataFrame as input:
data = pd.read_csv('yourDataset.csv')
result = encoder.encode(data)

# if label is given in the dataset
result = encoder.encode(data, label='name_of_label_column')

```

### TODO:
1. Deal with ID-like columns: identify (also let user decide?) and delete ? 
2. Find better way to distinguish categorical columns.
3. More functionality and more flexible implementation for user to config prefered setting.


## Discretizer
Take a column (pandas Series) as input, output a column with discretized values. For the discretize() function:
* **by**: "width": discretize by equal width; "frequency": discretize by equal frequency; "kmeans": discretize by kmeans clustering; "gmm": discretize by Gaussian mixure models clustering. default by="width".
* **num_bins**: number of bins. default num_bins=10.
* **labels**: list of values for the discretized bins, currently only for binning methods where orders of values are kept (by width and by frequency). default labels= [0,1,2...].


Note, currently: 
* Missing cells remain missing in the output column.

### Usage:
```python
from dsbox.datapreprocessing.cleaner import discretizer

data = pd.read_csv('yourDataset.csv')
col = data["column_name"]
# 10 bins, discretize by equal width
result = discretizer.discretize(col)
# 5 bins, discretize by gmm
result = discretizer.discretize(col,num_bins=5,by='gmm')
# or you can replace original column in the dataset with discretized values
data["column_name"] = result

```

### TODO:
- See if a better k, number of bins to choose can be found automatically. e.g. num_bins='auto'.

## Missing value imputer
This component is for missing value imputation. This module is designed to support:

1. multiple ways to impute data, including our self-defined methods.
2. missing pattern related analysis (to be exposed)

Now the functionality is limited to:

* one label problem

### Dependencies
[check here](environment_py3.yml)

if you have conda, simply do the following:

```sh
conda-env create .
source activate mv-py3
```

### Usage:
see [test_examples](./test_examples):

```python
"""
sample program for classification problem
"""
def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])

import pandas as pd

from dsbox.datapreprocessing.cleaner import GreedyImputation

# STEP 1: get data
data_path = "../../dsbox-data/o_38/encoded/"
data_name = data_path + "trainData_encoded.csv"
label_name = data_path + "trainTargets_encoded.csv" # make sure your label target is in the second column of this file

data = pd.read_csv(data_name)
label = text2int(pd.read_csv(label_name)["Class"])

data.drop("d3mIndex",axis=1)    # drop because id, useless

# STEP 2: go to use the Imputer !
# check GreedyImputation
imputer = GreedyImputation(verbose=1)
imputer.set_training_data(inputs=data, outputs=label)	# unsupervised
imputer.fit(timeout=10)	# give 10 seconds to fit
print (imputer.get_call_metadata())	# to see wether fit worked
result = imputer.produce(inputs=data, timeout=0.01)
print (imputer.get_call_metadata())	# to see wether produce worked

```



### methods
see [methods](methods.md) for details.

## One-hot encoder
The encoder takes pandas DataFrame as input, then one-hot encode columns which are considered categorical. 

```
class Encoder(categorical_features='95in10')
```

For **categorical_features = '95in10'**, it takes a column as category if:
* its dtype is not float and
* 95% of its data fall in 10 values.
* For the rest values (not top 10) with low frequency, put into one column _[colname]\_other\__

Note: 
* Maximum number of values encoded: **n_limit**, Whether to convert other text columns to integers: **text2int**.
* Apply set_params() function to change the two parameters' values. 
* For one-hot encoded columns, in the output there would always be a _[colname]\_other__ column for values not appear in fitted data and values with fewer occurrence (when there are more than **n_limit** distinct values).


### Usage:
```python
from dsbox.datapreprocessing.cleaner import Encoder

train_x = pd.read_csv(train_dataset)
test_x = pd.read_csv(test_dataset)

enc = Encoder()
enc.set_training_data(inputs=train_x)
enc.fit()
result = enc.produce(inputs=train_x)

p = enc.get_params()
enc2 = Encoder()
enc2.set_params(params=p)
result2 = enc2.produce(inputs=test_x)
```

### TODO:
1. Find better way to distinguish categorical columns - by statistics?
2. More functionality and more flexible implementation for user to config prefered setting.
3. Deal with ID-like columns: identify (also let user decide?) and delete ? 


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



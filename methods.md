# Imputer
## pre-conditions

1. has missing values
2. all numerical values  

take Pandas.DataFrame as data input. Internally using Numpy Array to do the imputations.

## Imputation method
### 1. baseline:

- 1.1.  drop all rows that contains missing value
- 1.2.  drop all columns that contains missing value
- 1.3.  using `mean` to impute all  

### 2. Greedy search 

from left to right, search for **best** combination of imputation. supported imputation method:

- mean
- min
- max
- zero

$best$ means the following: `machine learning model` & `metrics`
give out the best performance.

### 3. iterative regession
say input data has `n` dimentions, `m` in that has missing value. 

```
1. init all missing values with `mean` of the column
2. loop until the imputed values converge:
	for each_col in m:
		 remove the imputed values in each_col, then using all other
		 features to impute it (a regression model trained and fit
		 here)
```

in reality, the loop is set to `30`, an empirical number.

### 4. MICE
now using the [fancyimpute-mice](https://github.com/hammerlab/fancyimpute/blob/master/fancyimpute/mice.py), a relative complex method...

their method is like:

1. outer layer: get multiple imputation results (the number is a hyper parameter), and take the average as final imputated values (there is also a hyperparameter to determine how many result to be averaged, from last one).
2. for each imputation, they using: 
	1. init all with `mean` or `median` 
	2. to impute each column, only chose `n_nearest_columns` of this column (e.g using correlation+randomness to chose) to use
	3. then impute the missing value, there are two options, by default "col" is used, which is [Posterior Predictive Distribution](https://www.cs.utah.edu/~fletcher/cs6957/lectures/BayesianLinearRegression.pdf); also can be "pmm"
	
	then go for loops to run...


**diff btween iterative_regre**: using `n_nearest_columns `, useful when number of column is huge; using `Posterior Predictive Distribution`.

### 5. knn
now using the [fancyimpute-knn](https://github.com/hammerlab/fancyimpute/blob/master/fancyimpute), they also trigger [knnimpute](https://github.com/hammerlab/knnimpute)

according to their `readme`: 
> Nearest neighbor imputations which weights samples using the mean squared difference on features for which two rows both have observed data.

because it will calculate the weights using mean square error, this method requires each feature of inpute data are scaled (mean 0 and variance 1).


## Evaluation
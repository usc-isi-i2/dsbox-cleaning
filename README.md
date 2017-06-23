# Introduction
This component is for missing value imputation. It will give the evaluation result of different imputation method. Now the functionality is limited to:

* only classification problem
* one label problem and label target is the second column in label file

### Dependencies
1. python 2.7
2. pandas >= 0.20.1
3. dsbox-datapreprocessing

### Usage:
see test_example.py



## Evaluation
### Imputation method
1. baseline:
	- 1.1 drop all rows that contains missing value
	- 1.2. drop all columns that contains missing value
2. Greedy search for best combination of imputation. supported imputation method:
	- mean
	- min
	- max
	- zero
	- new value (for categorical column)

### machine learning model
for each imputation method, evaluated by the best score from the following machine learning models:

1. linear SVM
2. Logistic Regression CV
3. Decision Tree
	
	
## Data Handling
### 1. remove useless columns
1. remove the id-like column (user specified)
2. remove the empty column

### 2. convert categorical to indicator
### 3. imputation based conversion
Ideally, the data now should all be numeric values, but contain some missing values. Then the data will be fed into different imputation method and be fixed the missing values. 

The outputed data should contains no missing value (numpy array format) and can be directly solved by machine learning models (except the baseline1.1 method, details come in next)

### 4. learning & predicting
the input data in this module is expected as numpy array format and already with imputeted missing values. If contains missing value, the corresponding rows will be removed and counted as "cannot predict".



## pipeline design
dataPrep -> mainTest -> baseline, imputationGreedy

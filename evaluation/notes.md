dataset: 28 total = 11 regression + 17 classification, all with missing values, more or less.

In the evaluation, everything is fixed (how split the data, machine learning model, etc.) except the different Imputation methods. So we use the machine learning model's performance to judge the performance of the Imputation method.

### how to generate result
edit and run the `evaluation.py` under `MVImputation` folder


### special treatment

1. encoder part still contains error...
	
	I add a self-defined simple preprocessing, to drop the non-sense column. In `evaluation.py`


2. if using linearRegression() for regression problem, all task will be negtive perfomance using r2_score. So change to SVR
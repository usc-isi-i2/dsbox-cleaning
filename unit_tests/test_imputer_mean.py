"""
test program for mean, a TransformerPrimitive imputer
"""
import sys
sys.path.append("../")
def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])

import pandas as pd

from dsbox.datapreprocessing.cleaner import MeanImputation
from primitive_interfaces.base import CallMetadata

# get data
data_name =  "data.csv"
label_name =  "targets.csv" # make sure your label target is in the second column of this file
data = pd.read_csv(data_name, index_col='d3mIndex')
missing_value_mask = pd.isnull(data)
label = text2int(pd.read_csv(label_name, index_col='d3mIndex')["Class"])


import unittest

class TestMean(unittest.TestCase):

	def setUp(self):
		self.imputer = MeanImputation(verbose=1)
		self.enough_time = 100
		self.not_enough_time = 0.000001

	def test_init(self):
		self.assertEqual(self.imputer.get_call_metadata(), 
			CallMetadata(has_finished=False, iterations_done=False))

	def test_run(self):
		# part 1
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data)	
		imputer.fit(timeout=self.enough_time)
		print (imputer.get_params())
		self.assertEqual(imputer.get_call_metadata(), 
			CallMetadata(has_finished=True, iterations_done=True))

		result = imputer.produce(inputs=data, timeout=self.enough_time)
		self.helper_impute_result_check(data, result)

		# part2: test set_params()
		imputer2 = MeanImputation(verbose=1)
		imputer2.set_params(params=imputer.get_params())
		self.assertEqual(imputer.get_call_metadata(), 
			CallMetadata(has_finished=True, iterations_done=True))
		result2 = imputer2.produce(inputs=data, timeout=self.enough_time)
		self.assertEqual(result2.equals(result), True)	# two imputers' results should be same
		self.assertEqual(imputer.get_call_metadata(), 
			CallMetadata(has_finished=True, iterations_done=True))

	# mean imputation is too fast to make it timeout

	# def test_timeout(self):
	# 	imputer = MeanImputation(verbose=1)
	# 	imputer.set_training_data(inputs=data)	
	# 	imputer.fit(timeout=self.not_enough_time)
	# 	self.assertEqual(imputer.get_call_metadata(), 
	# 		CallMetadata(has_finished=False, iterations_done=False))
	# 	with self.assertRaises(ValueError):	# ValueError is because: have on fitted yet
	# 		result = imputer.produce(inputs=data, timeout=self.not_enough_time)

	def test_noMV(self):
		"""
		test on the dataset has no missing values
		"""
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data)	
		imputer.fit(timeout=self.enough_time)
		result = imputer.produce(inputs=data, timeout=self.enough_time)
		result2 = imputer.produce(inputs=result, timeout=self.enough_time)	# `result` contains no missing value

		self.assertEqual(result.equals(result2), True)
	
	def test_notAlign(self):
		"""
		test the case that the missing value situations in trainset and testset are not aligned. eg:
			`a` missing-value columns in trainset, `b` missing-value columns in testset.
			`a` > `b`, or `a` < `b`
		"""
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data)	
		imputer.fit(timeout=self.enough_time)
		result = imputer.produce(inputs=data, timeout=self.enough_time)
		# PART1: when `a` > `b`
		data2 = result.copy()
		data2["T3"] = data["T3"].copy()	# only set this column to original column, with missing vlaues
		result2 = imputer.produce(inputs=data2, timeout=self.enough_time)
		self.helper_impute_result_check(data2, result2)

		# PART2: when `a` < `b`
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data2)	
		imputer.fit(timeout=self.enough_time)
		result = imputer.produce(inputs=data, timeout=self.enough_time)	
		# data contains more missingvalue columns than data2, 
		# the imputer should triger default impute method for the column that not is trained
		self.helper_impute_result_check(data, result)

		# PART3: trunk the data : sample wise
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data)	
		imputer.fit(timeout=self.enough_time)
		result = imputer.produce(inputs=data[0:20], timeout=self.enough_time)
		self.helper_impute_result_check(data[0:20], result)


	def helper_impute_result_check(self, data, result):
		"""
		check if the imputed reuslt valid
		now, check for:
		1. contains no nan anymore
		2. orignal non-nan value should remain the same
		"""
		# check 1
		self.assertEqual(pd.isnull(result).sum().sum(), 0)

		# check 2
		# the original non-missing values must keep unchanged
		# to check, cannot use pd equals, since the imputer may convert:
		# 1 -> 1.0
		# have to do loop checking
		missing_value_mask = pd.isnull(data)
		for col_name in data:
			data_non_missing = data[~missing_value_mask[col_name]][col_name]
			result_non_missing = result[~missing_value_mask[col_name]][col_name]
			for i in data_non_missing.index:
				self.assertEqual(data_non_missing[i]==result_non_missing[i], True, 
					msg="not equals in column: {}".format(col_name))

if __name__ == '__main__':
    unittest.main()
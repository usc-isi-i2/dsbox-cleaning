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
data_path = "../../../dsbox-data/o_38/encoded/"
data_name = data_path + "trainData_encoded.csv"
label_name = data_path + "trainTargets_encoded.csv" # make sure your label target is in the second column of this file
data = pd.read_csv(data_name)
label = text2int(pd.read_csv(label_name)["Class"])
data = data.drop("d3mIndex",axis=1)    # drop because id, useless


import unittest

class TestMean(unittest.TestCase):

	def setUp(self):
		self.imputer = MeanImputation(verbose=1)
		self.enough_time = 100
		self.not_enough_time = 0.00001

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

	def test_timeout(self):
		imputer = MeanImputation(verbose=1)
		imputer.set_training_data(inputs=data)	
		imputer.fit(timeout=self.not_enough_time)
		self.assertEqual(imputer.get_call_metadata(), 
			CallMetadata(has_finished=False, iterations_done=False))
		with self.assertRaises(ValueError):	# ValueError is because: have on fitted yet
			result = imputer.produce(inputs=data, timeout=self.not_enough_time)

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
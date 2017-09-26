#!/usr/bin/env python3

from os import path

import pandas as pd
import numpy as np
import json

from dsbox.datapreprocessing.cleaner import GreedyImputation
from dsbox.datapreprocessing.cleaner import Encoder
from dsbox.datapreprocessing.cleaner.encoder import Params

from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder

# Example for the documentation of the TA1 pipeline submission process
#
# It executes a TA1 pipeline using a ta1-pipeline-config.json file that follows this structure:
# {
#   "problem_schema":"path/to/problem_schema.json",
#   "dataset_schema":"path/to/dataset_schema.json",
#   "data_root":"path/to/data/root/folder/",
#   "output_file":"path/to/output/file"
# }

#============= helper function==============
def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])


# Load the json configuration file
with open("ta1-pipeline-config.json", 'r') as inputFile:
    jsonCall = json.load(inputFile)
    inputFile.close()

# Load the json dataset description file
with open(jsonCall['dataset_schema'], 'r') as inputFile:
    datasetSchema = json.load(inputFile)
    inputFile.close()

# Load the input files from the data_root folder path information, replacing missing values with zeros
dataRoot = jsonCall['data_root']
trainData = pd.read_csv( path.join(dataRoot, 'trainData.csv.gz') )
trainTargets = pd.read_csv( path.join(dataRoot, 'trainTargets.csv.gz') )
testData = pd.read_csv( path.join(dataRoot, 'testData.csv.gz') )

# print(trainData.head())
# print(trainTargets.head())
# print(np.asarray(trainTargets['Class']))
# print(testData.head())

enc = Encoder()
enc.set_training_data(inputs=trainData)
enc.fit()
encodedData = enc.produce(inputs=trainData)
encodedTestData = enc.produce(inputs=testData)

# Initialize the DSBox imputer
imputer = GreedyImputation(verbose=0)
imputer.set_training_data(inputs=encodedData, outputs=text2int(trainTargets['Class']))	# unsupervised
imputer.fit(timeout=100.0)	# give 10 seconds to fit

print (imputer.get_call_metadata())	# to see wether fit worked
print('\nParams:')
print(imputer.get_params())

imputer2 = GreedyImputation(verbose=0)
imputer2.set_params(params=imputer.get_params())

imputedData = imputer2.produce(inputs=encodedData, timeout=100.0)
print (imputer2.get_call_metadata())	# to see wether produce worked


model = BaggingClassifier()
trainedModel = model.fit(imputedData, np.asarray(trainTargets['Class']))


predictedTargets = trainedModel.predict(imputer.produce(inputs=encodedTestData))
print(predictedTargets)

# Outputs the predicted targets in the location specified in the JSON configuration file
with open(jsonCall['output_file'], 'w') as outputFile:
    output = pd.DataFrame(predictedTargets).to_csv(outputFile, index_label='d3mIndex', header=['Class'])

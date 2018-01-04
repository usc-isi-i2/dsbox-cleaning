#!/usr/bin/env python3

from os import path
# TODO: remove relative import once all installed
import sys
sys.path.append("../")
#================

import pandas as pd
import numpy as np
import json


from dsbox.datapreprocessing.cleaner import MICE, MiceHyperparameter
from dsbox.datapreprocessing.cleaner import Encoder, EncHyperparameter

from d3m_metadata import params

from sklearn.ensemble import BaggingClassifier

# Example for the documentation of the TA1 pipeline submission process
#
# It executes a TA1 pipeline using a ta1-pipeline-config.json file that follows this structure:
# {
#    "train_data":"path/to/train/data/folder/",
#    "test_data":"path/to/test/data/folder/",
#    "output_folder":"path/to/output/folder/"
# }

# Load the json configuration file
with open("ta1-pipeline-config.json", 'r') as inputFile:
    jsonCall = json.load(inputFile)
    inputFile.close()

# Load the problem description schema
with open( path.join(jsonCall['train_data'], 'problem_TRAIN', 'problemDoc.json' ) , 'r') as inputFile:
    problemSchema = json.load(inputFile)
    inputFile.close()

# Load the json dataset description file
with open( path.join(jsonCall['train_data'], 'dataset_TRAIN', 'datasetDoc.json' ) , 'r') as inputFile:
    datasetSchema = json.load(inputFile)
    inputFile.close()

# Get the target and attribute column ids from the dataset schema for training data
trainAttributesColumnIds = [ item['colIndex'] for item in datasetSchema['dataResources'][0]['columns'] if 'attribute' in item['role'] ]
trainTargetsColumnIds = [ item['colIndex'] for item in problemSchema['inputs']['data'][0]['targets'] ]

# Exit if more than one target
if len(trainTargetsColumnIds) > 1:
    print('More than one target in the problem. Exiting.')
    exit(1)

# Get the attribute column ids from the problem schema for test data (in this example, they are the same)
testAttributesColumnIds = trainAttributesColumnIds

# Load the tabular data file for training, replace missing values, and split it in train data and targets
trainDataResourcesPath = path.join(jsonCall['train_data'], 'dataset_TRAIN', datasetSchema['dataResources'][0]['resPath'])
trainData = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainAttributesColumnIds)
trainTargets = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainTargetsColumnIds)

# Load the tabular data file for training, replace missing values, and split it in train data and targets
testDataResourcesPath = path.join(jsonCall['test_data'], 'dataset_TEST', datasetSchema['dataResources'][0]['resPath'])
testData = pd.read_csv( testDataResourcesPath, header=0, usecols=testAttributesColumnIds)

# Get the d3mIndex of the testData
d3mIndex = pd.read_csv( testDataResourcesPath, header=0, usecols=['d3mIndex'])

print(trainData.head())
print(trainTargets.head())
print(np.asarray(trainTargets['Class']))
print(testData.head())


hp = EncHyperparameter.sample()
enc = Encoder(hyperparams=hp)
enc.set_training_data(inputs=trainData)
enc.fit()
encodedData = enc.produce(inputs=trainData).value
encodedTestData = enc.produce(inputs=testData).value

# Initialize the DSBox imputer
hp = MiceHyperparameter.sample()
imputer = MICE(hyperparams=hp)
imputedData = imputer.produce(inputs=encodedData, timeout=100).value

model = BaggingClassifier()
trainedModel = model.fit(imputedData, np.asarray(trainTargets['Class']))


predictedTargets = trainedModel.predict(imputer.produce(inputs=encodedTestData).value)

# Append the d3mindex column to the predicted targets
predictedTargets = pd.DataFrame({'d3mIndex':d3mIndex['d3mIndex'], 'Class':predictedTargets})
print(predictedTargets.head())

# Get the file path of the expected outputs
outputFilePath = path.join(jsonCall['output_folder'], problemSchema['expectedOutputs']['predictionsFile'])

# Outputs the predicted targets in the location specified in the JSON configuration file
with open(outputFilePath, 'w') as outputFile:
    output = predictedTargets.to_csv(outputFile, index=False, columns=['d3mIndex', 'Class'])

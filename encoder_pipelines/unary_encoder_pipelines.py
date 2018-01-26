#!/usr/bin/env python3

from os import path

import sys
sys.path.append("../")

import pandas as pd 
import numpy as np
import json

from dsbox.datapreprocessing.cleaner import UnaryEncoder, UEncHyperparameter
from dsbox.datapreprocessing.profiler import category_detection

from sklearn.preprocessing import Imputer
from sklearn.ensemble import BaggingClassifier

# ta1-pipeline-config.json structure:
# {
#   "train_data":"path/to/train/data/folder/",
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
target_name = problemSchema['inputs']['data'][0]['targets'][0]['colName']

print(trainData.head())
print(trainTargets.head())
print(np.asarray(trainTargets[target_name]))
print(testData.head())

# Categorical detection
category = category_detection.category_detect(trainData)
columns = trainData.columns
inttype = [int,np.int8,np.int16, np.int32, np.int64,float]
encode_columns = [i for i in range(len(trainData.columns)) if category[columns[i]] and trainData[columns[i]].dtype in inttype]

# Initialize the DSBox Unary Encoder
hp = UEncHyperparameter(text2int=True,targetColumns=encode_columns)
enc = UnaryEncoder(hyperparams=hp)
enc.set_training_data(inputs=trainData)
enc.fit()

print(type(enc.get_params()))
print(enc.get_params())

imputer = Imputer()
model = BaggingClassifier()


encodedTrainData = enc.produce(inputs=trainData).value
processedTrainData = imputer.fit_transform(encodedTrainData)
trainedModel = model.fit(processedTrainData, np.asarray(trainTargets[target_name]))

print(encodedTrainData.columns) # encoded result


predictedTargets = trainedModel.predict(imputer.fit_transform(enc.produce(inputs=testData).value))

# Append the d3mindex column to the predicted targets
predictedTargets = pd.DataFrame({'d3mIndex':d3mIndex['d3mIndex'], target_name:predictedTargets})
print(predictedTargets.head())

# Get the file path of the expected outputs
outputFilePath = path.join(jsonCall['output_folder'], problemSchema['expectedOutputs']['predictionsFile'])


# Outputs the predicted targets in the location specified in the JSON configuration file
with open(outputFilePath, 'w') as outputFile:
    output = predictedTargets.to_csv(outputFile, index=False, columns=['d3mIndex', target_name])

#!/usr/bin/env python3

from os import path

import pandas as pd
import numpy as np
import json

from dsbox.datapreprocessing.cleaner import Encoder, EncHyperparameter
from dsbox.datapreprocessing.cleaner.encoder import Params

from sklearn.preprocessing import Imputer
from sklearn.ensemble import BaggingClassifier

# ta1-pipeline-config.json structure:
# {
#   "problem_schema":"path/to/problem_schema.json",
#   "dataset_schema":"path/to/dataset_schema.json",
#   "data_root":"path/to/data/root/folder/",
#   "output_file":"path/to/output/file"
# }

# Load the json configuration file
with open("ta1-pipeline-config.json", 'r') as inputFile:
    jsonCall = json.load(inputFile)
    inputFile.close()

# Load the json dataset description file
with open(jsonCall['dataset_schema'], 'r') as inputFile:
    datasetSchema = json.load(inputFile)
    inputFile.close()

# Load the input files from the data_root folder path information
dataRoot = jsonCall['data_root']
trainData = pd.read_csv( path.join(dataRoot, 'trainData.csv.gz') )
trainTargets = pd.read_csv( path.join(dataRoot, 'trainTargets.csv.gz') )
testData = pd.read_csv( path.join(dataRoot, 'testData.csv.gz') )

print(trainData.head())
print(trainTargets.head())
print(np.asarray(trainTargets['Class']))
print(testData.head())

# Initialize the DSBox Encoder

hp = EncHyperparameter.sample()
enc = Encoder(hp)
enc.set_training_data(inputs=trainData)
enc.fit()

print(type(enc.get_params()))
print(enc.get_params())

imputer = Imputer()
model = BaggingClassifier()

print(trainData.columns)

encodedTrainData = enc.produce(inputs=trainData)
processedTrainData = imputer.fit_transform(encodedTrainData)
trainedModel = model.fit(processedTrainData, np.asarray(trainTargets['Class']))

print(encodedTrainData.columns) # encoded result


predictedTargets = trainedModel.predict(imputer.fit_transform(enc.produce(inputs=testData)))
print(predictedTargets)

# Outputs the predicted targets in the location specified in the JSON configuration file
with open(jsonCall['output_file'], 'w') as outputFile:
    output = pd.DataFrame(predictedTargets).to_csv(outputFile, index_label='d3mIndex', header=['Class'])

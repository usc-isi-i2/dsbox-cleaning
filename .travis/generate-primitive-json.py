#!/usr/bin/env python3

# import argparse
import os.path
import subprocess

from dsbox.datapreprocessing.cleaner import config as cleaner_config

# from dsbox_dev_setup import path_setup
# path_setup()

import dsbox

# parser = argparse.ArgumentParser(
#     description='Generate primitive.json descriptions')
# parser.add_argument(
#     'dirname', action='store', help='Top-level directory to store the json descriptions')
# arguments = parser.parse_args()

PREFIX = 'd3m.primitives.'
PRIMITIVES = [(p, cleaner_config) for p in [
    'data_cleaning.CleaningFeaturizer.DSBOX',
    'data_preprocessing.Encoder.DSBOX',
    'data_preprocessing.UnaryEncoder.DSBOX',
    'data_preprocessing.GreedyImputation.DSBOX',
    'data_preprocessing.IterativeRegressionImputation.DSBOX',
    'data_preprocessing.MeanImputation.DSBOX',
    'normalization.IQRScaler.DSBOX',
    'data_cleaning.Labeler.DSBOX',
    'normalization.Denormalize.DSBOX',
    'schema_discovery.Profiler.DSBOX',
    'data_cleaning.FoldColumns.DSBOX',
    'data_preprocessing.VerticalConcat.DSBOX',
    'data_preprocessing.EnsembleVoting.DSBOX',
    'data_preprocessing.Unfold.DSBOX',
    'data_preprocessing.Splitter.DSBOX',
    'data_preprocessing.HorizontalConcat.DSBOX',
    'data_augmentation.Augmentation.DSBOX',
    'data_augmentation.QueryDataframe.DSBOX',
    'data_augmentation.Join.DSBOX',
    'data_transformation.ToNumeric.DSBOX'
]
]
dirname = "output"
for p, config in PRIMITIVES:
    print('Generating json for primitive ' + p)
    primitive_name = PREFIX + p
    outdir = os.path.join(dirname, 'v' + config.D3M_API_VERSION,
                          config.D3M_PERFORMER_TEAM, primitive_name,
                          config.VERSION)
    subprocess.run(['mkdir', '-p', outdir])

    json_filename = os.path.join(outdir, 'primitive.json')
    print('    at ' + json_filename)
    command = ['python', '-m', 'd3m.index',
               'describe', '-i', '4', primitive_name]
    with open(json_filename, 'w') as out:
        subprocess.run(command, stdout=out)

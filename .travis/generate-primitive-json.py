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
    'data_cleaning.cleaning_featurizer.DSBOX',
    'data_preprocessing.encoder.DSBOX',
    'data_preprocessing.unary_encoder.DSBOX',
    'data_preprocessing.greedy_imputation.DSBOX',
    'data_preprocessing.iterative_regression_imputation.DSBOX',
    'data_preprocessing.mean_imputation.DSBOX',
    'normalization.iqr_scaler.DSBOX',
    'data_cleaning.labeler.DSBOX',
    'normalization.denormalize.DSBOX',
    'schema_discovery.profiler.DSBOX',
    'data_cleaning.column_fold.DSBOX',
    'data_preprocessing.vertical_concat.DSBOX',
    'data_preprocessing.ensemble_voting.DSBOX',
    'data_preprocessing.unfold.DSBOX',
    'data_preprocessing.splitter.DSBOX',
    'data_preprocessing.horizontal_concat.DSBOX',
    'data_augmentation.data_augmentation.datamart_augmentation.DSBOX',
    'data_augmentation.datamart_query.DSBOX',
    'data_augmentation.datamart_join.DSBOX',
    'data_transformation.to_numeric.DSBOX'
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

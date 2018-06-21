#!/usr/bin/env python3

import argparse
import os.path
import subprocess

from dsbox.datapreprocessing.cleaner import config as cleaner_config

# from dsbox_dev_setup import path_setup
# path_setup()

import dsbox

parser = argparse.ArgumentParser(description='Generate primitive.json descriptions')
parser.add_argument(
    'dirname', action='store', help='Top-level directory to store the json descriptions')
arguments = parser.parse_args()

PREFIX = 'd3m.primitives.dsbox.'
PRIMITIVES = [(p, cleaner_config) for p in [
    'MeanImputation',
    'IterativeRegressionImputation',
    'GreedyImputation',
    'MiceImputation',
    'KnnImputation',
    'Encoder',
    'UnaryEncoder',
    'IQRScaler']
              ]

for p, config in PRIMITIVES:
    print('Generating json for primitive ' + p)
    primitive_name = PREFIX + p
    outdir = os.path.join(arguments.dirname, 'v' + config.D3M_API_VERSION,
                          config.D3M_PERFORMER_TEAM, primitive_name,
                          config.VERSION)
    subprocess.run(['mkdir', '-p', outdir])

    json_filename = os.path.join(outdir, 'primitive.json')
    print('    at ' + json_filename)
    command = ['python', '-m', 'd3m.index', 'describe', '-i', '4', primitive_name]
    with open(json_filename, 'w') as out:
        subprocess.run(command, stdout=out)

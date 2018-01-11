#!/usr/bin/env python3

import argparse
import os.path
import subprocess

from dsbox_dev_setup import path_setup
path_setup()

import dsbox

parser = argparse.ArgumentParser(description='Generate primitive.json descriptions')
parser.add_argument(
    'dirname', action='store', help='Top-level directory to store the json descriptions')
arguments = parser.parse_args()


PREFIX = 'd3m.primitives.dsbox.'
PRIMITIVES = [
    'MeanImputation', 
    'IterativeRegressionImputation', 
    'GreedyImputation',
    'MiceImputation',
    'KnnImputation',
    'Encoder',
    'UnaryEncoder'
]

for p in PRIMITIVES:
    print('Generating json for primitive ' + p)
    primitive_name = PREFIX + p
    outdir = os.path.join(arguments.dirname, 'v'+dsbox.__d3m_api_version__, 
                       dsbox.__d3m_performer_team__, primitive_name, 
                       'v'+dsbox.__version__)
    subprocess.run(['mkdir', '-p', outdir])

    json_filename =  os.path.join(outdir, 'primitive.json')
    command = ['python', '-m', 'd3m.index', 'describe', '-i', '4', primitive_name]
    with open(json_filename, 'w') as out:
        subprocess.run(command, stdout=out)

#!/usr/bin/env python3

import argparse
import os.path
import subprocess
from dsbox.datapreprocessing.cleaner import config as cleaner_config

parser = argparse.ArgumentParser(description='Generate primitive.json descriptions')
parser.add_argument('--dirname', action='store', default = 'output', help='Top-level directory to store the json descriptions, i.e. primitives_repo directory')
arguments = parser.parse_args()

PREFIX = 'd3m.primitives.'
PRIMITIVES = []

try:
    f = open("../setup.py","r")
except:
    f = open("setup.py","r")

line = f.readline()
while line:
    line = f.readline()
    if "'d3m.primitives': [" in line:
        line = f.readline()
        while line and "]," not in line:
            try:
                primitive_name = line.split("=")[0]
                primitive_name = primitive_name.split("'")[1]
                if primitive_name[-1] == " ":
                    primitive_name = primitive_name[:-1]
                PRIMITIVES.append(primitive_name)
                line = f.readline()
            except:
                break
        break
f.close()

for p in PRIMITIVES:
    print('Generating json for primitive ' + p)
    primitive_name = PREFIX + p
    outdir = os.path.join(dirname, 'v' + cleaner_config.D3M_API_VERSION,
                          cleaner_config.D3M_PERFORMER_TEAM, primitive_name,
                          cleaner_config.VERSION)
    subprocess.run(['mkdir', '-p', outdir])

    json_filename = os.path.join(outdir, 'primitive.json')
    print('    at ' + json_filename)
    command = ['python', '-m', 'd3m.index',
               'describe', '-i', '4', primitive_name]
    with open(json_filename, 'w') as out:
        subprocess.run(command, stdout=out)

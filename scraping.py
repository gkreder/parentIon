################################################################################
# Gabe Reder - gkreder@gmail.com
################################################################################
import sys
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pyteomics import mzml, mgf
import argparse
################################################################################

################################################################################
parser = argparse.ArgumentParser()

# Required file and parameter inputs 
parser.add_argument('--in_file', required = True)
parser.add_argument('--in_fname_col', required = True)
parser.add_argument('--out_fname_col', required = True)
parser.add_argument('--index_col', required = True)

parser.add_argument('--title_col', default = None)
parser.add_argument('--collision_energy_col', default = None)

args = parser.parse_args()
################################################################################
# 
################################################################################
if args.in_file.endswith(".tsv"):
    df = pd.read_csv(args.in_file, sep = '\t')
elif args.in_file.endswith(".xlsx"):
    df = pd.read_excel(args.in_file)
else:
    sys.exit(f"Error - unrecognized input format for file {args.in_file}")

prev_inFile = ""
df = df.sort_values(by = args.in_fname_col)
for i, fname_in in enumerate(tqdm(df[args.in_fname_col])):
    fname_out = df[args.out_fname_col].values[i]
    sid = df[args.index_col].values[i]
    if fname_in != prev_inFile:
        f = mzml.MzML(fname_in)
    prev_inFile = fname_in
    spec = f.get_by_index(sid)

    rt = spec['scanList']['scan'][0]['scan start time']
    array = list(zip(spec['m/z array'], spec['intensity array']))
    pepmass = str(spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
    charge = str(int(spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']))

    if args.title_col == None:
        title = spec['spectrum title']
    else:
        title = df[args.title_col].values[i]

    if args.collision_energy_col == None:
        collision_energy = spec['precursorList']['precursor'][0]['activation']['collision energy']
    else:
        collision_energy = df[args.collision_energy_col].values[i]

    fout = open(fname_out, 'w')
    print('BEGIN IONS', file = fout)
    print('PEPMASS=' + pepmass, file = fout)
    print('CHARGE=' + charge, file = fout)
    print('TITLE=' + title, file = fout)
    print('RT=' + str(rt), file = fout)
    print('', file = fout)
    for mz, inten in array:
        print(mz, inten, file = fout)
    print('', file = fout)
    print('END IONS\n', file = fout)
    fout.close()





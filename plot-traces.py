import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import argv
import pathlib

p = pathlib.Path(argv[1])
p_files  = [x for x in p.iterdir() if x.is_file()]
n_files  = len(p_files)
npzfiles = '{LOG_DIR}/%04d.npz'.format(LOG_DIR=argv[1])

mdp_cost = np.zeros(n_files, dtype=np.float32)

for i in range(n_files):
    trace = np.load(npzfiles%i)
    mdp_cost
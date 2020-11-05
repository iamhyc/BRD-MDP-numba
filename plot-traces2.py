#!/usr/bin/env python3
import sys, glob
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from params import BETA, LQ

# customize matplotlib plotting
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)

def getAverageCost(ref, start, end):
    pass

def getAverageNumber(ref, start, end):
    pass

try:
    _, log_folder, log_type  = sys.argv
    glob.glob( Path(log_folder, log_type) )
except Exception as e:
    print('Loading traces failed with:', sys.argv)
    raise e
finally:
    pass


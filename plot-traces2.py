#!/usr/bin/env python3
import glob, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from params import BETA, LQ

# customize matplotlib plotting
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)

# globally load trace folder
try:
    _, log_type, log_folder = sys.argv
    #TODO: load what trace?
except Exception:
    print('Loading traces failed with:', sys.argv)
    pass


# main section
if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        raise e
    finally:
        pass

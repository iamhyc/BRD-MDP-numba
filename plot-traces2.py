#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from params import BETA, LQ
from numba import njit, prange
# customize matplotlib plotting
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)
# tag the benchmarks
#["MDP", "Selfish", "QAware", "Random"]
# global variables
global records_path

def getStatistics(ref, start, end):
    # 'MDP_value', 'MDP_ap_stat', 'MDP_es_stat', 'MDP_admissions', 'MDP_departures'
    # 'Selfish_ap_stat', 'Selfish_es_stat', 'Selfish_admissions', 'Selfish_departures'
    # 'QAware_ap_stat', 'QAware_es_stat', 'QAware_admissions', 'QAware_departures'
    # 'Random_ap_stat', 'Random_es_Stat', 'Random_admissions', 'Random_departures'

    # average number, average cost, discounted cost;
    # average JCT
    pass

def plot_statistics():
    records_path
    pass

try:
    _, log_folder, log_type  = sys.argv
    records_path = sorted( Path(log_folder).glob(log_type+'-*') )
    plot_statistics()
except Exception as e:
    print('Loading traces failed with:', sys.argv)
    raise e
finally:
    pass


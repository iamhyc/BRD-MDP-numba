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

def getAverageNumber(ref, start, end):
    # self.acc_num / self.timeslot
    pass

def getAverageCost(ref, start, end):
    # self.acc_cost / self.timeslot
    pass

def getDiscountedCost(ref, start, end):
    pass

def getAverageJCT(ref, start, end):
    # self.acc_cost / self.acc_arr
    pass

def getAverageThroughput(ref, start, end):
    # self.acc_dep / self.acc_arr
    pass

def getStatistics(ref, start, end):
    # 'MDP_ap_stat', 'MDP_es_stat', 'MDP_admissions', 'MDP_departures'
    # 'Selfish_ap_stat', 'Selfish_es_stat', 'Selfish_admissions', 'Selfish_departures'
    # 'QAware_ap_stat', 'QAware_es_stat', 'QAware_admissions', 'QAware_departures'
    # 'Random_ap_stat', 'Random_es_Stat', 'Random_admissions', 'Random_departures'
    pass

def plot_statistics():
    statistics = list()
    for record in records_path:
        #TODO: iterDir loading, and pass to GET functions
        #TODO: compose statistics dict, and push into list
        pass
    pass

try:
    _, log_folder, log_type  = sys.argv
    records_path = sorted( Path(log_folder).glob(log_type+'-*') )
except Exception as e:
    print('Loading traces failed with:', sys.argv)
    raise e
finally:
    pass


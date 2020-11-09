#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from params import BETA, LQ, N_SLT
from numba import njit, prange
# customize matplotlib plotting
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)
# tag the records
DATA_TAG = ['ap_stat', 'es_stat', 'admissions', 'departures']
ALG_TAG  = ['MDP', 'Selfish', 'QAware', 'Random']
get_tag  = lambda y:[x+'_'+y for x in ALG_TAG]
# global variables
global records_path

def getAverageNumber(ref, start=0, end=-1):
    # acc_num / timeslot
    _weakref = ref[start:end]
    acc_num = np.zeros((len(ALG_TAG),), dtype=np.int32)
    time_slot = len(_weakref) * N_SLT
    for sample in _weakref:
        acc_num += list(map( lambda x: sample[x], get_tag('ap_stat') ))
        sample[ _tag ]
        get_tag('es_stat')
        pass
    pass

def getAverageCost(ref, start=0, end=-1):
    # self.acc_cost / self.timeslot
    pass

def getDiscountedCost(ref, start=0, end=-1):
    pass

def getAverageJCT(ref, start, end=-1):
    # self.acc_cost / self.acc_arr
    pass

def getAverageThroughput(ref, start=0, end=-1):
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
    for record_dir in records_path:
        record = sorted( record_dir.iterdir() )
        record = [np.load(x) for x in record]
        statistics.append({
            'AverageNumber' : getAverageNumber(record),
            'AverageCost'   : getAverageCost(record),
            'DiscountedCost': getDiscountedCost(record) 
        })
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


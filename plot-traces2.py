#!/usr/bin/env python3
import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from params import BETA, GAMMA, LQ, N_SLT
from numba import njit, prange
# customize matplotlib plotting
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif':['Helvetica']})
# rc('text', usetex=True)
# tag the records
DATA_TAG = ['ap_stat', 'es_stat', 'admissions', 'departures']
ALG_TAG  = ['MDP', 'Tight', 'Selfish', 'QAware', 'Random']
ALG_COLOR= ['r',   'k',     'b',       'g',      'c']
get_tag  = lambda y:[x+'_'+y for x in ALG_TAG]
# global variables
global records_path

# acc_num / time_slot
def getAverageNumber(ref, start=0, end=-1):
    _weakref = ref[start:end]
    acc_num  = np.zeros((len(ALG_TAG),), dtype=np.int32)
    time_slot = len(_weakref) * N_SLT
    for sample in _weakref:
        acc_num += np.array([ sample[x].sum() for x in get_tag('ap_stat') ])
        acc_num += np.array([ sample[x].sum() for x in get_tag('es_stat') ])
        pass
    return acc_num / time_slot

# acc_cost / time_slot
def getAverageCost(ref, start=0, end=-1):
    _weakref = ref[start:end]
    acc_cost = np.zeros((len(ALG_TAG),), dtype=np.int32)
    time_slot = len(_weakref) * N_SLT
    for sample in _weakref:
        # np.sum(ap_stat) + np.sum(es_stat) + _penalty
        acc_cost += np.array([ sample[x].sum() for x in get_tag('ap_stat') ])
        acc_cost += np.array([ sample[x].sum() for x in get_tag('es_stat') ])
        acc_cost += np.array([ BETA*np.count_nonzero(x==LQ) for x in get_tag('es_stat') ])
        pass
    return acc_cost / time_slot

# disc_cost / time_slot
def getDiscountedCost(ref, start=0, end=-1):
    _weakref = ref[start:end]
    disc_cost = np.zeros((len(ALG_TAG),), dtype=np.float32)
    time_slot = len(_weakref) * N_SLT
    for idx, sample in enumerate(_weakref):
        _cost = np.zeros((len(ALG_TAG),), dtype=np.float32)
        _cost += np.array([ sample[x].sum() for x in get_tag('ap_stat') ])
        _cost += np.array([ sample[x].sum() for x in get_tag('es_stat') ])
        _cost += np.array([ BETA*np.count_nonzero(x==LQ) for x in get_tag('es_stat') ])
        disc_cost += pow(GAMMA, idx) * _cost
        pass
    return disc_cost / time_slot

# acc_cost / acc_arr
def getAverageJCT(ref, start=0, end=-1):
    # self.acc_cost / self.acc_arr
    _weakref = ref[start:end]
    avg_cost = getAverageCost(ref, start, end)
    acc_cost = avg_cost * len(_weakref) * N_SLT
    acc_arr  = np.array([ (_weakref[-1][x]-_weakref[0][x]).sum() for x in get_tag('admissions') ])
    return acc_cost / acc_arr

# acc_dep / acc_arr
def getAverageThroughput(ref, start=0, end=-1):
    _weakref = ref[start:end]
    acc_dep  = np.array([ (_weakref[-1][x]-_weakref[0][x]).sum() for x in get_tag('departures') ])
    acc_dep += np.array([ _weakref[-1][x].sum() for x in get_tag('es_stat') ]) # blame remaining jobs as departures
    acc_arr  = np.array([ (_weakref[-1][x]-_weakref[0][x]).sum() for x in get_tag('admissions') ])
    return acc_dep / acc_arr

def load_statistics():
    statistics = list()
    for record_dir in tqdm(records_path, desc='Loading statistics'):
        _save_file = save_path.joinpath( record_dir.stem+'.npz' )
        if _save_file.exists():
            statistics.append( np.load(_save_file) )
        else:
            record = sorted( record_dir.iterdir() )
            record = [np.load(x) for x in record]
            _result = {
                'AverageNumber' : getAverageNumber(record),
                'AverageCost'   : getAverageCost(record),
                'DiscountedCost': getDiscountedCost(record),
                'AverageJCT'    : getAverageJCT(record),
                'AverageThroughput': getAverageThroughput(record)
            }
            statistics.append(_result)
            np.savez_compressed(_save_file.as_posix(), **_result)
        pass
    #
    fig, ax = plt.subplots()
    _sum = np.zeros((len(ALG_TAG),), dtype=np.float32)
    for idx, data in enumerate(statistics):
        _sum += data['AverageCost']
        ax.plot(idx, _sum[0]/(idx+1), '.r')
    plt.show()
    pass

try:
    _, log_folder, log_type  = sys.argv
    records_path = sorted( Path(log_folder).glob(log_type+'-*') )
    save_path    = Path(log_folder, log_type+'_statistics')
    save_path.mkdir(exist_ok=True)
    load_statistics()
    #
    
except Exception as e:
    print('Loading traces failed with:', sys.argv)
    raise e
finally:
    pass


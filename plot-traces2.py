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
ALG_TAG  = ['MDP', 'Tight']
ALG_COLOR= ['r',   'k'    ]
ALG_NUM  = len(ALG_TAG)
get_tag  = lambda y:[x+'_'+y for x in ALG_TAG]
# global variables
global records_path
from params import EVAL_RANGE

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)
    pass

# acc_num / time_slot
def getAverageNumber(ref, start=0, end=-1):
    _weakref = ref[start:end]
    acc_num  = np.zeros((len(ALG_TAG),), dtype=np.int32)
    time_slot = len(_weakref)
    for sample in _weakref:
        acc_num += np.array([ sample[x].sum() for x in get_tag('ap_stat') ])
        acc_num += np.array([ sample[x].sum() for x in get_tag('es_stat') ])
        pass
    return acc_num / time_slot

# acc_cost / time_slot
def getAverageCost(ref, start=0, end=-1):
    _weakref = ref[start:end]
    acc_cost = np.zeros((len(ALG_TAG),), dtype=np.int32)
    time_slot = len(_weakref)
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
    time_slot = len(_weakref)
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
    acc_cost = avg_cost * len(_weakref)
    acc_arr  = np.array([ (_weakref[-1][x]-_weakref[0][x]).sum() for x in get_tag('admissions') ])
    return acc_cost / acc_arr

# acc_dep / acc_arr
def getAverageThroughput(ref, start=0, end=-1):
    _weakref = ref[start:end]
    prev_arr = np.zeros(ALG_NUM) if start==0 else ref[start-1]
    acc_dep  = np.array([ (_weakref[-1][x]-_weakref[0][x]).sum() for x in get_tag('departures') ])
    acc_dep += np.array([ _weakref[-1][x].sum() for x in get_tag('es_stat') ]) # blame remaining jobs as departures
    acc_arr  = np.array([ (_weakref[-1][x]-prev_arr).sum() for x in get_tag('admissions') ])
    return acc_dep / acc_arr

def load_statistics(ti_num):
    _pattern = 'ti{num}-*'.format(num=ti_num)
    records_path = sorted( Path(log_folder).glob( _pattern ) )
    save_path    = Path(log_folder, 'ti{num}_statistics'.format(num=ti_num))
    save_path.mkdir(exist_ok=True)

    samples = list()
    for record_dir in tqdm(records_path, desc='Loading statistics-%r'%ti_num):
        _save_file = save_path.joinpath( record_dir.stem+'.npz' )
        if _save_file.exists():
            samples.append( np.load(_save_file) )
        else:
            record = sorted( record_dir.iterdir() )
            record = [np.load(x) for x in record]
            _result = {
                'AverageNumber' : getAverageNumber(record),
                'AverageCost'   : getAverageCost(record),
                'DiscountedCost': getDiscountedCost(record),
                # 'AverageJCT'    : getAverageJCT(record),
                # 'AverageThroughput': getAverageThroughput(record)
            }
            samples.append(_result)
            np.savez_compressed(_save_file.as_posix(), **_result)
        pass
    return samples

def plot_statistics():
    _sum = np.zeros((len(ALG_TAG),), dtype=np.float32)
    for t,sample in enumerate(statistics):
        _sum += sample['DiscountedCost']
        for idx in range(ALG_NUM):
            plt.plot((t+1), _sum[idx]/(t+1), ALG_COLOR[idx]+'.')
    plt.title('Discounted Cost')
    plt.xlabel('Index of Broadcast Interval')
    plt.legend(ALG_TAG)
    plt.show()
    pass

#---------------------------------------------------------------------------------------#
def plot_bar_graph():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    plot_alg = ['MDP', 'Selfish', 'QAware', 'Random']
    x_range = np.arange(4)
    # Average Cost
    average_cost = np.zeros((len(ALG_TAG),), dtype=np.float32)
    for sample in statistics:
        average_cost += sample['AverageCost']
    average_cost = average_cost / len(statistics)
    average_cost = [average_cost[ALG_TAG.index(x)] for x in plot_alg]
    bar_plot1 = ax1.bar(x_range, average_cost, edgecolor='black', color='#1F77B4')
    [bar_plot1[i].set_hatch(x) for i,x in enumerate(['.', '/', 'x', '\\'])]
    ax1.set_title('(a)', y=-0.075, fontsize=20)
    ax1.set_xticklabels(['']+plot_alg, fontsize=14)
    ax1.set_ylabel('Average Cost', fontsize=16)
    ax1.yaxis.set_label_coords(-0.15,0.5)
    autolabel(ax1, bar_plot1)

    # Average Job Response Time
    average_JCT = np.zeros((len(ALG_TAG),), dtype=np.float32)
    for sample in statistics:
        average_JCT += sample['AverageJCT']
    average_JCT = average_JCT / len(statistics)
    average_JCT = [average_JCT[ALG_TAG.index(x)] for x in plot_alg]
    bar_plot2 = ax2.bar(x_range, average_JCT, edgecolor='black', color='#1F77B4')
    [bar_plot2[i].set_hatch(x) for i,x in enumerate(['.', '/', 'x', '\\'])]
    ax2.set_title('(b)', y=-0.075, fontsize=20)
    ax2.set_xticklabels(['']+plot_alg, fontsize=14)
    ax2.set_ylabel('Average Job Response Time', fontsize=16)
    ax2.yaxis.set_label_coords(-0.14,0.5)
    autolabel(ax2, bar_plot2)

    #Average Job Dropping Rate
    average_throughput = np.zeros((len(ALG_TAG),), dtype=np.float32)
    for sample in statistics:
        average_throughput += sample['AverageThroughput']
    average_throughput = average_throughput / len(statistics)
    average_throughput = [average_throughput[ALG_TAG.index(x)] for x in plot_alg]
    average_throughput = 1.0 - np.array(average_throughput)
    # average_throughput+= [0.001, 0, 0.001, 0.001]
    bar_plot3 = ax3.bar(x_range, average_throughput, edgecolor='black', color='#1F77B4')
    [bar_plot3[i].set_hatch(x) for i,x in enumerate(['.', '/', 'x', '\\'])]
    ax3.set_ylim([0.0, 0.05])
    ax3.set_title('(c)', y=-0.075, fontsize=20)
    ax3.set_xticklabels(['']+plot_alg, fontsize=14)
    ax3.set_ylabel('Average Job Dropping Rate', fontsize=16)
    ax3.yaxis.set_label_coords(-0.150,0.5)
    autolabel(ax3, bar_plot3)

    plt.show()
    pass

def plot_tight_bound():
    r_mdp, r_ti = list(), list()
    for rng in EVAL_RANGE:
        _sum = np.array([0, 0], dtype=np.float32)
        for sample in statistics[rng]:
            _sum += sample['DiscountedCost'] #AverageNumber/AverageCost/DiscountedCost
        _sum = N_SLT*_sum/len(statistics[rng])
        r_mdp.append(_sum[0])
        r_ti.append(_sum[1])
        pass

    # plt.plot(EVAL_RANGE, r_mdp, '.r-')
    # plt.plot(EVAL_RANGE, r_ti,  '.b-')
    plt.plot(EVAL_RANGE, np.array(r_ti)-np.array(r_mdp), '.k-')
    plt.show()
    pass

try:
    _, log_folder, _  = sys.argv
    statistics = dict()
    for rng in EVAL_RANGE:
        statistics.update( {rng : load_statistics(rng)} )
    # plot_statistics()
    # Fig. 5. Illustration of performance metrics comparison with benchmarks.
    # plot_bar_graph()
    # Fig. 7 Illustration of monotonical performance gap decreasing.
    plot_tight_bound()
except Exception as e:
    print('Loading traces failed with:', sys.argv)
    raise e
finally:
    pass


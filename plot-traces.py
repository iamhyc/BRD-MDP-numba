#!/usr/bin/env python3
import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import argv
import glob

npzfiles = glob.glob( '{LOG_DIR}/*.npz'.format(LOG_DIR=argv[1]) )
npzfiles.sort()
n_files  = len(npzfiles)

MDP_trace     = list()
Selfish_trace = list()
QAware_trace  = list()
Random_trace  = list()

for i,_file in enumerate(npzfiles):
    trace = np.load(_file)
    MDP_trace.append({
        'ap_stat': trace['MDP_ap_stat'],
        'es_stat': trace['MDP_es_stat'],
        'value'  : trace['MDP_value']
    })
    QAware_trace.append({
        'ap_stat': trace['QAware_ap_stat'],
        'es_stat': trace['QAware_es_stat'],
    })
    Random_trace.append({
        'ap_stat': trace['Random_ap_stat'],
        'es_stat': trace['Random_es_Stat'],
    })
    Selfish_trace.append({
        'ap_stat': trace['Selfish_ap_stat'],
        'es_stat': trace['Selfish_es_stat'],
    })
    pass

def plot_bar_graph():
    pass

def plot_cost_vs_time():
    MDP_cost     = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in MDP_trace]
    QAware_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in QAware_trace]
    Random_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Random_trace]
    Selfish_cost = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Selfish_trace]

    plt.plot(range(n_files), MDP_cost,     '-ro')
    plt.plot(range(n_files), QAware_cost,  '-go')
    plt.plot(range(n_files), Random_cost,  '-co')
    plt.plot(range(n_files), Selfish_cost, '-bo')

    plt.legend(['MDP Policy', 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.show()
    pass

def plot_cost_cdf_vs_time():
    y = [0] * 4
    y[0] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in MDP_trace])
    y[1] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in QAware_trace])
    y[2] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Random_trace])
    y[3] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Selfish_trace])
    
    ylim = max([arr.max() for arr in y])
    pmf_x = np.linspace(0, ylim, num=1000)
    pmf_y = np.zeros((5, 1000))
    # pmf_inter_x     = np.linspace(0, ylim, num=200)
    # pmf_inter_y = np.zeros((5, 200))

    for i in range(4):
        for j in range(1,1000):
            pmf_y[i][j] = np.logical_and( y[i]>=pmf_x[j-1], y[i]<pmf_x[j] ).sum()
        pmf_y[i] = np.cumsum(pmf_y[i]) / 1000
        # interp_func = interp1d(x, pmf_y[i], kind='quadratic')
        # pmf_inter_y[i] = interp_func(pmf_inter_x)
        pass

    plt.xlim(0, ylim)
    plt.plot(pmf_x, pmf_y[0], '-r')
    plt.plot(pmf_x, pmf_y[1], '-g')
    plt.plot(pmf_x, pmf_y[2], '-c')
    plt.plot(pmf_x, pmf_y[3], '-b')

    plt.legend(['MDP Policy', 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.show()
    pass

# plot_bar_graph()
plot_cost_vs_time()
# plot_cost_cdf_vs_time()

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

mdp_trace     = list()
selfish_trace = list()
sqf_trace     = list()
random_trace  = list()

for i,_file in enumerate(npzfiles):
    trace = np.load(_file)
    mdp_trace.append({
        'ap_stat': trace['mdp_ap_stat'],
        'es_stat': trace['mdp_es_stat'],
        'value'  : trace['mdp_value']
    })
    sqf_trace.append({
        'ap_stat': trace['sqf_ap_stat'],
        'es_stat': trace['sqf_es_stat'],
    })
    random_trace.append({
        'ap_stat': trace['random_ap_stat'],
        'es_stat': trace['random_es_Stat'],
    })
    selfish_trace.append({
        'ap_stat': trace['selfish_ap_stat'],
        'es_stat': trace['selfish_es_stat'],
    })
    pass

def plot_cost_vs_time():
    mdp_cost     = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in mdp_trace]
    sqf_cost     = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in sqf_trace]
    random_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in random_trace]
    selfish_cost = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in selfish_trace]

    plt.plot(range(n_files), mdp_cost,     '-ro')
    plt.plot(range(n_files), sqf_cost,     '-go')
    plt.plot(range(n_files), random_cost,  '-co')
    plt.plot(range(n_files), selfish_cost, '-bo')

    plt.legend(['MDP Policy', 'SQF Policy', 'Random Policy', 'Selfish Policy'])
    plt.show()
    pass

def plot_cost_cdf_vs_time():
    y = [0] * 4
    y[0] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in mdp_trace])
    y[1] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in sqf_trace])
    y[2] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in random_trace])
    y[3] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in selfish_trace])
    
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

    plt.legend(['MDP Policy', 'SQF Policy', 'Random Policy', 'Selfish Policy'])
    plt.show()
    pass

# plot_cost_vs_time()
plot_cost_cdf_vs_time()
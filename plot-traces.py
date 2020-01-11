#!/usr/bin/env python3
import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import argv
import glob
from params import BETA,LQ

MDP_LABEL = 'MDP Policy'

log_dir  = argv[1]
npzfiles = glob.glob('{LOG_DIR}/*.npz'.format(LOG_DIR=log_dir))
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

def autolabel(bar_plot, labels):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.,
                height+0.15,
                '%.2f'%labels[idx],
                ha='center', va='bottom', rotation=0)

def getCost(ap_stat, es_stat):
    _penalty = BETA * np.count_nonzero( es_stat[:,:,0]==LQ )
    return _penalty + np.sum(ap_stat) + np.sum(es_stat[:,:,0])

def plot_bar_graph():
    summary_file = '{LOG_DIR}/summary'.format(LOG_DIR=log_dir)
    summary = np.load(summary_file)

    x = np.arange(4)

    plt.subplot(1, 3, 1)
    average_cost = [summary['MDP_average_cost'],
                    summary['Selfish_average_cost'],
                    summary['QAware_average_cost'],
                    summary['Random_average_cost']]
    bar_plot = plt.bar(x, average_cost, color='#1F77B4')
    autolabel(bar_plot, average_cost)
    plt.title('(a)', y=-0.075)
    plt.xticks(x, ['MDP', 'Selfish', 'Queue-aware', 'Random'])
    plt.ylabel('Average Cost')

    plt.subplot(1, 3, 2)
    average_JCT = [summary['MDP_average_JCT'],
                    summary['Selfish_average_JCT'],
                    summary['QAware_average_JCT'],
                    summary['Random_average_JCT']]
    bar_plot = plt.bar(x, average_JCT, color='#1F77B4')
    autolabel(bar_plot, average_JCT)
    plt.title('(b)', y=-0.075)
    plt.xticks(x, ['MDP', 'Selfish', 'Queue-aware', 'Random'])
    plt.ylabel('Average JCT')

    plt.subplot(1, 3, 3)
    average_throughput = [summary['MDP_average_throughput'],
                    summary['Selfish_average_throughput'],
                    summary['QAware_average_throughput'],
                    summary['Random_average_throughput']]
    bar_plot = plt.bar(x, average_throughput, color='#1F77B4')
    autolabel(bar_plot, average_throughput)
    plt.title('(c)', y=-0.075)
    plt.xticks(x, ['MDP', 'Selfish', 'Queue-aware', 'Random'])
    plt.ylabel('Average Throughput')
    
    plt.show()
    pass

def plot_cost_vs_time():
    MDP_cost     = [getCost(x['ap_stat'], x['es_stat']) for x in MDP_trace]
    QAware_cost  = [getCost(x['ap_stat'], x['es_stat']) for x in QAware_trace]
    Random_cost  = [getCost(x['ap_stat'], x['es_stat']) for x in Random_trace]
    Selfish_cost = [getCost(x['ap_stat'], x['es_stat']) for x in Selfish_trace]

    plt.plot(range(n_files), MDP_cost,     '-ro')
    plt.plot(range(n_files), QAware_cost,  '-go')
    plt.plot(range(n_files), Random_cost,  '-co')
    plt.plot(range(n_files), Selfish_cost, '-bo')

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.show()
    pass

def plot_number_vs_time():
    MDP_cost     = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in MDP_trace]
    QAware_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in QAware_trace]
    Random_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Random_trace]
    Selfish_cost = [np.sum(x['ap_stat'])+np.sum(x['es_stat'][:,:,0]) for x in Selfish_trace]

    plt.plot(range(n_files), MDP_cost,     '-ro')
    plt.plot(range(n_files), QAware_cost,  '-go')
    plt.plot(range(n_files), Random_cost,  '-co')
    plt.plot(range(n_files), Selfish_cost, '-bo')

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.ylabel('Number of Jobs')
    plt.xlabel('Index of Time Slots')
    plt.show()
    pass

def plot_number_cdf_vs_time():
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

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.ylabel('CDF')
    plt.xlabel('Cost per Time Slot')
    plt.show()
    pass

def plot_cost_cdf_vs_time():
    y = [0] * 4
    y[0] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in MDP_trace])
    y[1] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in QAware_trace])
    y[2] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in Random_trace])
    y[3] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in Selfish_trace])
    
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

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'])
    plt.ylabel('CDF')
    plt.xlabel('Cost per Time Slot')
    plt.show()
    pass

plot_bar_graph()
# plot_number_vs_time()
# plot_cost_vs_time()
plot_number_cdf_vs_time()
# plot_cost_cdf_vs_time()

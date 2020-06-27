#!/usr/bin/env python3
import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import argv
from os.path import join as path_join
import glob
from parse import parse
from params import BETA,LQ

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']}) # Palatino font
rc('text', usetex=True)

MDP_LABEL  = 'MDP Policy'
CUT_NUM    = 100
LABEL_SIZE = 24
DISP_FLAG  = True

try:
    log_dir  = argv[1]
    log_num  = argv[1].split('-')[1]
    npzfiles = glob.glob('{LOG_DIR}/*.npz'.format(LOG_DIR=log_dir))
    npzfiles.sort()
    n_files  = len(npzfiles)
    NUM_DATA = n_files

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
except Exception as e:
    print('No arguments input.')

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

def getCost(ap_stat, es_stat):
    _penalty = BETA * np.count_nonzero( es_stat==LQ )
    return _penalty + np.sum(ap_stat) + np.sum(es_stat)

def plot_bar_graph():
    summary_file = '{LOG_DIR}/summary'.format(LOG_DIR=log_dir)
    summary = np.load(summary_file)

    x = np.arange(4)
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    average_cost = [summary['MDP_average_cost'],
                    summary['Selfish_average_cost'],
                    summary['QAware_average_cost'],
                    summary['Random_average_cost']]
    # average_cost = [137.08,177.15,169.81,219.36]
    bar_plot1 = ax1.bar(x, average_cost, color='#1F77B4')
    # ax1.set_ylim([100])
    ax1.set_title('(a)', y=-0.075, fontsize=20)
    ax1.set_xticklabels(['', 'MDP', 'Selfish', 'Queue-aware', 'Random'], fontsize=14)
    ax1.set_ylabel('Average Cost', fontsize=16)

    average_JCT = [summary['MDP_average_JCT'],
                    summary['Selfish_average_JCT'],
                    summary['QAware_average_JCT'],
                    summary['Random_average_JCT']]
    # average_JCT = [291.97,377.56,361.78,467.28]
    bar_plot2 = ax2.bar(x, average_JCT, color='#1F77B4')
    ax2.set_title('(b)', y=-0.075, fontsize=20)
    ax2.set_xticklabels(['','MDP', 'Selfish', 'Queue-aware', 'Random'], fontsize=14)
    ax2.set_ylabel('Average Job Response Time', fontsize=16)

    average_throughput = [summary['MDP_average_throughput'],
                    summary['Selfish_average_throughput'],
                    summary['QAware_average_throughput'],
                    summary['Random_average_throughput']]
    average_throughput = 1.0 - np.array(average_throughput)
    # average_throughput = np.array([0.6571764705882352, 0.6211764705882353, 0.7105767195767195, 0.710475073313783])
    bar_plot3 = ax3.bar(x, average_throughput, color='#1F77B4')
    # ax3.set_ylim([0.0, 1.0])
    ax3.set_title('(c)', y=-0.075, fontsize=20)
    ax3.set_xticklabels(['', 'MDP', 'Selfish', 'Queue-aware', 'Random'], fontsize=14)
    ax3.set_ylabel('Average Job Dropping Rate', fontsize=16)
    
    autolabel(ax1, bar_plot1) #average_cost
    autolabel(ax2, bar_plot2) #average_JCT
    autolabel(ax3, bar_plot3) #average_throughput

    plt.savefig(path_join('figures','%s_bar_graph.pdf'%log_num), format='pdf', dpi=300)
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_cost_vs_time():
    MDP_cost     = [getCost(x['ap_stat'], x['es_stat']) for x in MDP_trace][CUT_NUM:]
    QAware_cost  = [getCost(x['ap_stat'], x['es_stat']) for x in QAware_trace][CUT_NUM:]
    Random_cost  = [getCost(x['ap_stat'], x['es_stat']) for x in Random_trace][CUT_NUM:]
    Selfish_cost = [getCost(x['ap_stat'], x['es_stat']) for x in Selfish_trace][CUT_NUM:]
    
    plt.grid()
    plt.plot(range(n_files-CUT_NUM), MDP_cost,     '-ro')
    plt.plot(range(n_files-CUT_NUM), QAware_cost,  '-go')
    plt.plot(range(n_files-CUT_NUM), Random_cost,  '-co')
    plt.plot(range(n_files-CUT_NUM), Selfish_cost, '-bo')

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=14)
    plt.ylabel('Cost', fontsize=16)
    plt.xlabel('Index of Broadcast Interval', fontsize=16)

    plt.savefig(path_join('figures','%s_cost_vs_time.pdf'%log_num), format='pdf', dpi=300)
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_number_vs_time():
    MDP_cost     = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in MDP_trace][CUT_NUM:]
    QAware_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in QAware_trace][CUT_NUM:]
    Random_cost  = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in Random_trace][CUT_NUM:]
    Selfish_cost = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in Selfish_trace][CUT_NUM:]

    fig, axes = plt.subplots()
    axes.grid()
    axes.plot(range(n_files-CUT_NUM), MDP_cost,     '-ro')
    axes.plot(range(n_files-CUT_NUM), QAware_cost,  '-go')
    axes.plot(range(n_files-CUT_NUM), Random_cost,  '-co')
    axes.plot(range(n_files-CUT_NUM), Selfish_cost, '-bo')

    axes.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=20)
    axes.set_ylabel('Number of Jobs in System', fontsize=24)
    axes.set_xlabel('Index of Broadcast Interval', fontsize=24)
    [tick.label.set_fontsize(24) for tick in axes.xaxis.get_major_ticks()]
    [tick.label.set_fontsize(24) for tick in axes.yaxis.get_major_ticks()]

    plt.savefig(path_join('figures','%s_number_vs_time.pdf'%log_num), format='pdf', dpi=300)
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_update_vs_time(serial, parallel):
    n1 = glob.glob(path_join(serial, '*.npz'))
    n1.sort()
    n2 = glob.glob(path_join(parallel, '*.npz'))
    n2.sort()
    n_files  = len(n1)

    y1_trace, y2_trace = list(), list()
    for i,_file in enumerate(n1):
        trace = np.load(_file)
        y1_trace.append({
            'ap_stat': trace['MDP_ap_stat'],
            'es_stat': trace['MDP_es_stat'],
            'value'  : trace['MDP_value']
        })
    for i,_file in enumerate(n2):
        trace = np.load(_file)
        y2_trace.append({
            'ap_stat': trace['MDP_ap_stat'],
            'es_stat': trace['MDP_es_stat'],
            'value'  : trace['MDP_value']
        })
    
    # getNumber
    # y1 = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in y1_trace][CUT_NUM:]
    # y2 = [np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in y2_trace][CUT_NUM:]
    # getCost
    y1 = [getCost(x['ap_stat'], x['es_stat']) for x in y1_trace][CUT_NUM:]
    y2 = [getCost(x['ap_stat'], x['es_stat']) for x in y2_trace][CUT_NUM:]

    fig, axes = plt.subplots()
    axes.grid()
    axes.plot(range(n_files-CUT_NUM), y1, '-ko')
    axes.plot(range(n_files-CUT_NUM), y2, '-ro')
    axes.legend(['Serial Update', 'Parallel Update'], fontsize=20)
    axes.set_ylabel('Number of Jobs in System', fontsize=24)
    axes.set_xlabel('Index of Broadcast Interval', fontsize=24)
    [tick.label.set_fontsize(24) for tick in axes.xaxis.get_major_ticks()]
    [tick.label.set_fontsize(24) for tick in axes.yaxis.get_major_ticks()]

    # plt.savefig(path_join('figures','%s_update.pdf'%log_num), format='pdf')
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_number_cdf_vs_time():
    y = [0] * 4
    y[0] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in MDP_trace][CUT_NUM:])
    y[1] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in QAware_trace][CUT_NUM:])
    y[2] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in Random_trace][CUT_NUM:])
    y[3] = np.sort([np.sum(x['ap_stat'])+np.sum(x['es_stat']) for x in Selfish_trace][CUT_NUM:])
    
    ylim = max([arr.max() for arr in y])
    pmf_x = np.linspace(0, ylim, num=NUM_DATA)
    pmf_y = np.zeros((5, NUM_DATA))
    # pmf_inter_x     = np.linspace(0, ylim, num=200)
    # pmf_inter_y = np.zeros((5, 200))

    for i in range(4):
        for j in range(1,NUM_DATA):
            pmf_y[i][j] = np.logical_and( y[i]>=pmf_x[j-1], y[i]<pmf_x[j] ).sum()
        pmf_y[i] = np.cumsum(pmf_y[i]) / NUM_DATA
        # interp_func = interp1d(x, pmf_y[i], kind='quadratic')
        # pmf_inter_y[i] = interp_func(pmf_inter_x)
        pass

    plt.grid()
    plt.xlim(0, ylim)
    plt.plot(pmf_x, pmf_y[0], '-r')
    plt.plot(pmf_x, pmf_y[1], '-g')
    plt.plot(pmf_x, pmf_y[2], '-c')
    plt.plot(pmf_x, pmf_y[3], '-b')

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=14)
    plt.ylabel('CDF', fontsize=16)
    plt.xlabel('Number per Broadcast Interval', fontsize=16)

    plt.savefig(path_join('figures','%s_num_cdf.pdf'%log_num), format='pdf')
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_cost_cdf_vs_time():
    y = [0] * 4
    y[0] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in MDP_trace][CUT_NUM:])
    y[1] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in QAware_trace][CUT_NUM:])
    y[2] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in Random_trace][CUT_NUM:])
    y[3] = np.sort([getCost(x['ap_stat'], x['es_stat']) for x in Selfish_trace][CUT_NUM:])
    
    ylim = max([arr.max() for arr in y])
    pmf_x = np.linspace(0, ylim, num=NUM_DATA)
    pmf_y = np.zeros((5, NUM_DATA))
    # pmf_inter_x     = np.linspace(0, ylim, num=200)
    # pmf_inter_y = np.zeros((5, 200))

    for i in range(4):
        for j in range(1,NUM_DATA):
            pmf_y[i][j] = np.logical_and( y[i]>=pmf_x[j-1], y[i]<pmf_x[j] ).sum()
        pmf_y[i] = np.cumsum(pmf_y[i]) / NUM_DATA
        # interp_func = interp1d(x, pmf_y[i], kind='quadratic')
        # pmf_inter_y[i] = interp_func(pmf_inter_x)
        pass

    plt.grid()
    plt.xlim(0, ylim)
    plt.plot(pmf_x, pmf_y[0], '-r')
    plt.plot(pmf_x, pmf_y[1], '-g')
    plt.plot(pmf_x, pmf_y[2], '-c')
    plt.plot(pmf_x, pmf_y[3], '-b')

    plt.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=14)
    plt.ylabel('CDF', fontsize=16)
    plt.xlabel('Cost per Broadcast Interval', fontsize=16)

    plt.savefig(path_join('figures','%s_cost_cdf.pdf'%log_num), format='pdf')
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

def plot_traces_study(prefix):
    fig, ax = plt.subplots(figsize=(8,6))
    _type   = prefix.split('-')[-2]
    traces  = glob.glob(prefix+'*')
    traces.sort()
    _format = prefix+'{}'
    x_ticks = [i for i,x in enumerate(traces)]
    x_ticklabels = [parse(_format,x)[0] for x in traces]

    mdp_cost, sf_cost, qf_cost, rd_cost = [], [], [], []
    for trace in traces:
        summary_file = '{}/summary'.format(trace)
        summary = np.load(summary_file)
        mdp_cost.append(summary['MDP_average_cost'])
        sf_cost.append(summary['Selfish_average_cost'])
        qf_cost.append(summary['QAware_average_cost'])
        rd_cost.append(summary['Random_average_cost'])
        pass

    ax.plot(x_ticks, mdp_cost, '-r^')
    ax.plot(x_ticks, qf_cost,  '-go')
    ax.plot(x_ticks, rd_cost,  '-cv')
    ax.plot(x_ticks, sf_cost,  '-bs')
    ax.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=14)
    ax.set_ylabel('Average Cost', fontsize=16)
    if _type=='arrival':
        ax.set_xlabel('Range of Expectation $\lambda_{m,j}$ of Arrival Rate', fontsize=16)
    elif _type=='proc':
        ax.set_xlabel('Range of Expectation $c_{m,j}$ of Processing Time Distribution', fontsize=16)
    else:
        ax.set_xlabel(_type, fontsize=16)

    ax.grid()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    pass

def myNumAPsPlot():
    fig, ax = plt.subplots(figsize=(8,6))

    x_ticks  = [3,4,5,6,7]
    mdp_cost = [29.47, 83.98,  120.19, 142.58, 151.60]
    sf_cost  = [81.77, 137.51, 151.37, 208.47, 230.31]
    qf_cost  = [30.81, 92.72,  139.87, 186.08, 212.33]
    rd_cost  = [32.87, 107.40, 178.48, 238.81, 277.99]

    ax.plot(x_ticks, mdp_cost, '-r^')
    ax.plot(x_ticks, qf_cost,  '-go')
    ax.plot(x_ticks, rd_cost,  '-cv')
    ax.plot(x_ticks, sf_cost,  '-bs')

    ax.legend([MDP_LABEL, 'Queue-aware Policy', 'Random Policy', 'Selfish Policy'], fontsize=14)
    ax.set_ylabel('Average Cost', fontsize=16)
    ax.set_xlabel('Number of APs', fontsize=16)

    ax.grid()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['3', '4', '5', '6', '7'])
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(path_join('figures','%s_ap_compare.pdf'%log_num), format='pdf')
    if DISP_FLAG: plt.show()
    plt.clf()
    pass

# plot_bar_graph()
# plot_number_vs_time()
# plot_cost_vs_time()
# plot_number_cdf_vs_time()
# plot_cost_cdf_vs_time()

# plot_update_vs_time(serial='traces-03896-serial', parallel='traces-03896-parallel')
plot_traces_study(argv[1])
# myNumAPsPlot()
# myProcDistPlot()
import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import argv
import pathlib

p = pathlib.Path(argv[1])
p_files  = [x for x in p.iterdir() if x.is_file()]
n_files  = len(p_files)
npzfiles = '{LOG_DIR}/%04d.npz'.format(LOG_DIR=argv[1])

mdp_trace = list()

for i in range(n_files):
    trace = np.load(npzfiles%i)
    mdp_trace.append({
        'ap_stat': trace['mdp_ap_stat'],
        'es_stat': trace['mdp_es_stat'],
        'value'  : trace['mdp_val']
    })
    pass

def plot_cost_vs_time():
    mdp_cost = np.sum(mdp_trace['ap_stat']) + np.sum(mdp_trace['es_stat'][:,:,0])
    plt.plot(range(n_files), mdp_cost, '-ro')
    plt.legend(['MDP Policy'])
    plt.show()

# plot_cost_vs_time()
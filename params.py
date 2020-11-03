#!/usr/bin/env python3 
import random
import numpy as np
from pathlib import *
from utility import *
from scipy.stats import norm
from termcolor import cprint
import networkx as nx
from pathlib import Path
import glob
from os import path
from sys import argv

TRACE_NUM   = 0
A_SCALE     = ['1_2x', '1_3x', '1_4x', '1_5x', '1_6x'][1]
TRACE_FOLDER='./data/trace-{:05d}-{}'.format(TRACE_NUM, A_SCALE)

MAP_SEED    = 3491
# RANDOM_SEED = random.randint(0, 2**16)
RANDOM_SEED = 3896
np.random.seed(RANDOM_SEED)

GAMMA   = 0.95
BETA    = 120
STAGE   = 150
STAGE_ALT = 120
STAGE_EVAL = [10]

N_AP  = 15
N_ES  = 10
N_JOB = 10
LQ    = 60 #maximum queue length on ES (inclusive)

TS    = 0.02         #timeslot, 20ms
TB    = 0.50         #interval, 500ms
N_SLT = int(TB/TS)   #25 slots/interval
N_CNT = 3*N_SLT + 1  #number of counters, ranged in [0,N_CNT-1]

BR_MIN     = int( 0.70 * N_SLT )    #(inclusive)
BR_MAX     = int( 0.90 * N_SLT )    #(exclusive)
BR_RNG     = np.arange(BR_MIN, BR_MAX,     step=1, dtype=np.int32)
BR_RNG_L   = len(BR_RNG)

UL_MIN     = int( 2.50 * N_SLT )    #(inclusive)
UL_MAX     = int( 3.00 * N_SLT )    #(exclusive)
UL_RNG     = np.arange(UL_MIN, UL_MAX+1,   step=1, dtype=np.int32)
UL_RNG_L   = len(UL_RNG)

PROC_MIN   = int( 1.10 * N_SLT ) #(inclusive)
PROC_MAX   = int( 1.20 * N_SLT ) #(inclusive)
PROC_RNG   = np.arange(PROC_MIN, PROC_MAX, step=1, dtype=np.int32)
PROC_RNG_L = len(PROC_RNG)
DIM_P      = (LQ+1)

GRAPH_RATIO = 0.3
U_FACTOR    = N_ES * (1/PROC_MAX) / N_AP

npzfile = 'logs/{:05d}.npz'.format(RANDOM_SEED)

def genProcessingParameter(es2ap_map, redo=False):
    global PROC_RNG, PROC_RNG_L
    if redo:
        PROC_RNG   = np.arange(PROC_MIN, PROC_MAX, step=1, dtype=np.int32)
        PROC_RNG_L = len(PROC_RNG)
    
    param = np.zeros((N_ES, N_JOB), dtype=np.int32)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            # _roll = np.random.randint(30)
            _tmp_dist = genHeavyHeadDist(PROC_RNG_L) #genHeavyTailDist(PROC_RNG_L) if _roll==0 else genHeavyHeadDist(PROC_RNG_L)
            param[m,j] = PROC_RNG[ multoss(_tmp_dist) ]
            if m==0: param[m,j] = param[m,j] / 6 #for cloud server computation time
    return param

def genDelayDistribution(redo=False):
    global BR_RNG, BR_RNG_L
    if redo:
        BR_RNG   = np.arange(BR_MIN, BR_MAX,     step=1, dtype=np.int32)
        BR_RNG_L = len(BR_RNG)
    
    dist = np.zeros((N_AP, BR_RNG_L), dtype=np.float64)
    for k in range(N_AP):
        dist[k] = genFlatDist(BR_RNG_L)
    return dist

def genUploadingProbabilities(es2ap_map):
    probs = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.float64)
    choice_dist = genFlatDist(UL_RNG_L)
    for j in range(N_JOB):
        for m in range(N_ES):
            for k in range(N_AP):
                if k==es2ap_map[m]: #co-location
                    probs[k,m,j] = np.ones(N_CNT, dtype=np.float64) #NOTE: uploaded at once (double-check needed)
                else: #follow some normal distribution
                    mean = UL_RNG[ multoss(choice_dist) ]
                    var  = (N_CNT - mean) / 3 #3-sigma-rule
                    rv   = norm(loc=mean, scale=var)
                    rv_total    = rv.cdf(N_CNT) - rv.cdf(range(N_CNT+1))
                    rv_prob     = np.diff( rv.cdf(range(N_CNT+1)) ) / rv_total[:-1]
                    probs[k,m,j] = rv_prob #true story, the last uploading is a ONE.
                    pass
    return probs

@njit()
def genTransitionMatrix():
    ul_mat  = np.zeros((N_AP,N_ES,N_JOB, N_CNT,N_CNT), dtype=np.float64)
    off_mat = np.zeros((N_AP,N_ES,N_JOB, N_CNT,N_CNT), dtype=np.float64)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            for k in prange(N_AP):
                ul_mat[k,m,j,   0, 0] = 1
                off_mat[k,m,j, -1,-1] = 1
                for i in prange(N_CNT-1):
                    ul_mat[k,m,j,  i,i+1] = 1 - ul_prob[k,m,j,i]
                    off_mat[k,m,j, i,i+1] =     ul_prob[k,m,j,i]
    return ul_mat, off_mat

def genConnectionMap():
    g = nx.barabasi_albert_graph(N_AP, 3, seed=42)
    bi_map = np.zeros((N_AP, N_ES), dtype=np.int32)

    np.random.seed( 1776 )
    es_nodes = np.sort( np.random.choice(range(N_AP), N_ES, replace=False) )
    for k in range(N_AP):
        _neighbors = g.neighbors(k)
        for idx,m in enumerate(es_nodes):
            bi_map[k,idx] = 1 if (m in _neighbors or m==k) else 0
    np.random.seed(RANDOM_SEED) #resume the seed

    return bi_map, es_nodes

def genMergedCandidateSet(bi_map):
    result = list()

    for k in range(N_AP):
        _candidate_set = list( np.where(bi_map[k] == 1)[0] )
        result.append([ set([k]), set(_candidate_set) ])

    tmp = np.ones(len(result))
    while np.count_nonzero(tmp):
        _len = len(result)
        tmp = np.zeros(_len)
        for i in range(_len-1):
            for j in range(i+1, _len):
                tmp[i] += 1 if (result[i][1] & result[j][1]==set()) else 0
        # print(result, tmp)
        
        if np.count_nonzero(tmp):
            i = np.where(tmp>0, tmp, np.inf).argmin()
            for j in range(_len):
                if not (result[i][1] & result[j][1]):
                    result[i][0] = result[i][0] | result[j][0]
                    result[i][1] = result[i][1] | result[j][1]
                    result.pop(j)
                    break
                pass
            pass
        pass

    return result

def loadArrivalTrace(index, loop=True):
    global TRACE_FOLDER
    trace_files = glob.glob( path.join(TRACE_FOLDER, '*.npy') )
    result = np.zeros((N_SLT,N_AP,N_JOB), dtype=np.int32)
    start_index = (N_SLT*index) % len(trace_files) # [start_idx, start_idx+N_SLT)
    for i in range(N_SLT):
        _idx = (start_index+i) % len(trace_files)
        result[i] = np.load( trace_files[_idx] )
    return result

#--------------------------- Execution Once When First Loaded ---------------------------#
try:
    assert( Path(npzfile).exists() ) #generate new params set only when no existing
    _params   = np.load(npzfile)
    arr_prob  = np.load(Path(TRACE_FOLDER, 'statistics'))
    # arr_prob  = _params['arr_prob']
    br_dist   = _params['br_dist']
    proc_mean = _params['proc_mean']
    ul_prob   = _params['ul_prob']
    ul_trans  = _params['ul_trans']
    off_trans = _params['off_trans']
    bi_map    = _params['bi_map']
    es2ap_map = _params['es2ap_map']
except AssertionError:
    print('Creating param file {:05d}.npz ...'.format(RANDOM_SEED))
    bi_map, es2ap_map = genConnectionMap()
    arr_prob  = np.load(Path(TRACE_FOLDER, 'statistics'))
    ul_prob   = genUploadingProbabilities(es2ap_map)
    br_dist   = genDelayDistribution()
    proc_mean = genProcessingParameter(es2ap_map)
    ul_trans, off_trans = genTransitionMatrix()

    np.savez(npzfile, **{
        'bi_map'   : bi_map,
        'es2ap_map': es2ap_map,
        'arr_prob' : arr_prob,
        'ul_prob'  : ul_prob,
        'br_dist'  : br_dist,
        'proc_mean': proc_mean,
        'ul_trans' : ul_trans,
        'off_trans': off_trans,
        'miscs'    : np.array([
            A_SCALE,MAP_SEED,RANDOM_SEED,       0.0,
            GAMMA,BETA,STAGE,                   0.0,
            N_AP,N_ES,N_JOB,LQ,                 0.0,
            TS,TB,N_SLT,N_CNT,                  0.0,
            BR_MIN,BR_MAX,BR_RNG_L,             0.0,
            UL_MIN,UL_MAX,UL_RNG_L,             0.0,
            PROC_MIN,PROC_MAX,PROC_RNG_L,DIM_P, 0.0,
        ])
    })
finally:
    br_dist   = genDelayDistribution()
    ul_rng    = np.arange(N_CNT, dtype=np.float64) #just facalited arrays
    #NOTE: inject external parameters via `online_main`
    if len(argv)>3 and argv[-2]=='--inject':
        exec(argv[-1])
    pass

#NOTE: generate subset partition
subset_map   = genMergedCandidateSet(bi_map)
N_SET        = len(subset_map)
subset_ind = np.zeros((N_SET, N_AP), dtype=np.int32)
for n in range(N_SET):
    for k in subset_map[n][0]:
        subset_ind[n, k] = 1
    pass
#NOTE: print subset partition
cprint('Subset Number: {}'.format(N_SET), 'red')
for item in subset_map:
    cprint(item, 'magenta')
print()
#NOTE: Test Code
# tmp_n, cnt = 100, 0
# while tmp_n>=7 and cnt<1000:
#     MAP_SEED = random.randint(0, 4096)
#     bi_map = genConnectionMap()
#     subset_map = genMergedCandidateSet(bi_map)
#     tmp_n = len(subset_map)
#     print(cnt, (tmp_n, MAP_SEED))
#     cnt += 1
#     pass
#----------------------------- End of First Loaded Section -----------------------------#

import random
import numpy as np
from pathlib import *
from utility import *
from scipy.stats import norm

RANDOM_SEED = random.randint(0, 2**16)
# RANDOM_SEED = 61937
np.random.seed(RANDOM_SEED)

GAMMA   = 0.95
BETA    = 20
STAGE   = 100

N_AP  = 10
N_ES  = 15
N_JOB = 10
LQ    = 100 #maximum queue length on ES (inclusive)

TS    = 0.02         #timeslot, 20ms
TB    = 0.50         #interval, 500ms
N_SLT = int(TB/TS)   #25 slots/interval
N_CNT = 3*N_SLT + 1  #number of counters, ranged in [0,N_CNT-1]

BR_MIN     = int( 0.50 * N_SLT )    #(inclusive)
BR_MAX     = int( 0.80 * N_SLT )    #(exclusive)
BR_RNG     = np.arange(BR_MIN, BR_MAX,       step=1, dtype=np.int32)
BR_RNG_L   = len(BR_RNG)

UL_MIN     = int( 1.50 * N_SLT )    #(inclusive)
UL_MAX     = int( 2.50 * N_SLT )    #(exclusive)
UL_RNG     = np.arange(UL_MIN, UL_MAX+1,     step=1, dtype=np.int32)
UL_RNG_L   = len(UL_RNG)

PROC_MIN   = int( 2.00 * N_SLT ) #(inclusive)
PROC_MAX   = int( 5.00 * N_SLT ) #(inclusive)
PROC_RNG_L = np.arange(PROC_MIN, PROC_MAX+1, step=1, dtype=np.int32)
DIM_P      = (LQ+1)

npzfile = 'logs/{:05d}.npz'.format(RANDOM_SEED)

@njit
def genProcessingParameter():
    dist = np.zeros((N_ES, N_JOB), dtype=np.float64)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            _roll = np.random.randint(2)
            _tmp_dist = genHeavyHeadDist(PROC_RNG_L) if _roll==0 else genHeavyTailDist(PROC_RNG_L) #1:1
            dist[m,j] = multoss(_tmp_dist) #get mean computation time
    return dist

@njit
def genBipartiteMap():
    bi_map = np.zeros((N_AP, N_ES), dtype=np.int32)
    
    return bi_map

def genDelayDistribution():
    dist = np.zeros((N_AP, BR_RNG_L), dtype=np.float64)
    for k in range(N_AP):
        dist[k] = genFlatDist(BR_RNG_L)
    return dist

def genUploadingProbabilities():
    probs = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.float64)
    choice_dist = genFlatDist(UL_RNG_L)
    for j in range(N_JOB):
        for m in range(N_ES):
            for k in range(N_AP):
                mean = UL_RNG[ multoss(choice_dist) ]
                var  = (N_CNT - mean) / 3 #3-sigma-rule
                rv   = norm(loc=mean, scale=var)
                rv_total    = rv.cdf(N_CNT) - rv.cdf(range(N_CNT+1))
                rv_prob     = np.diff( rv.cdf(range(N_CNT+1)) ) / rv_total[:-1]
                probs[k,m,j] = rv_prob #NOTE: true story, the last uploading is a ONE.
    return probs

@njit
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

if Path(npzfile).exists():
    _params   = np.load(npzfile)
    arr_prob  = _params['arr_prob']
    br_dist   = _params['br_dist'] #br_dist   = genDelayDistribution() # 
    proc_mean = _params['proc_mean']
    ul_prob   = _params['ul_prob']
    ul_trans  = _params['ul_trans']
    off_trans = _params['off_trans']
else:
    arr_prob  = 0.010 + 0.010 * np.random.rand(N_AP, N_JOB).astype(np.float64)
    ul_prob   = genUploadingProbabilities()
    br_dist   = genDelayDistribution()
    proc_mean = genProcessingParameter()
    ul_trans, off_trans = genTransitionMatrix()

    np.savez(npzfile, **{
        'arr_prob' : arr_prob,
        'ul_prob'  : ul_prob,
        'br_dist'  : br_dist,
        'proc_mean': proc_mean,
        'ul_trans' : ul_trans,
        'off_trans': off_trans,
        'miscs'    : np.array([
            N_AP,N_ES,N_JOB,LQ,
            TS,TB,N_SLT,N_CNT,
            BR_MIN,BR_MAX,BR_RNG_L,
            UL_MIN,UL_MAX,UL_RNG_L,
            PROC_MIN,PROC_MAX,PROC_RNG_L,
            DIM_P
        ])
    })
    pass

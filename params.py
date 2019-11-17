
import random
import numpy as numpy
from pathlib import *
from utility import *

RANDOM_SEED = random.randit(0, 2**16)
np.random.seed(RANDOM_SEED)

GAMMA = 0.90
BETA  = 10
STAGE = 1000

N_AP  = 5
N_ES  = 3
N_JOB = 10
LQ    = 10 #maximum queue length on ES (inclusive)

TS    = 0.01         #timeslot, 10ms
TB    = 0.50         #interval, 500ms
N_SLT = int(TB/TS)   #50 slots/interval
N_CNT = 3*N_SLT      #(for convenience)

BR_MIN   = int( 0.00 * N_SLT )    #(inclusive)
BR_MAX   = int( 0.50 * N_SLT )    #(exclusive)
BR_RNG   = np.arange(BR_MIN, BR_MAX,       step=1, dtype=np.int32)
BR_RNG_L = len(BR_RNG)

UL_MIN   = int( 0.00 * N_SLT )    #(inclusive)
UL_MAX   = int( 3.00 * N_SLT )    #(inclusive)NOTE: upper boundary included
UL_RNG   = np.arange(UL_MIN, UL_MAX+1,     step=1, dtype=np.int32)
UL_RNG_L = len(UL_RNG)
assert(N_CNT == UL_RNG_L))        #(for convenience)

PROC_MIN   = int( 1.00 * N_SLT )  #(inclusive)
PROC_MAX   = int( 3.50 * N_SLT )  #(exclusive) 
PROC_RNG   = np.arange(PROC_MIN, PROC_MAX, step=1, dtype=np.int32)
PROC_RNG_L = len(PROC_RNG)
DIM_P      = (LQ+1)*PROC_MAX

npzfile = 'logs/{:05d}.npz'.format(RANDOM_SEED)

@njit
def genProcessingDistribution():
    dist = np.zeros((N_ES, N_JOB, PROC_RNG_L), dtype=np.float32)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            _roll = np.random.randint(2)
            dist[m,j] = genHeavyHeadDist(PROC_RNG_L) if _roll==1 else genHeavyTailDist(PROC_RNG_L)
            # dist[m,j] = genHeavyHeadDist(PROC_RNG_L)
            # dist[m,j] = genHeavyTailDist(PROC_RNG_L)
            # dist[m,j] = genGaussianDist(PROC_RNG_L)
            # dist[m,j] = genSplitDist(PROC_RNG_L)
            # dist[m,j] = genFlatDist(PROC_RNG_L)
    return dist

@njit
def genUploadingDistribution():
    dist = np.zeros((N_AP, N_ES, N_JOB, UL_RNG_L), dtype=np.float32)
    #TODO:
    return dist

@njit
def genDelayDistribution():
    dist = np.zeros((), dtype=np.float32)
    #TODO:
    return dist

@njit
def genTransitionMatrix(ul_prob):
    ul_mat  = np.zeros((N_AP,N_ES,N_JOB), dtype=np.float32)
    off_mat = np.zeros((N_AP,N_ES,N_JOB), dtype=np.float32)
    #TODO:
    return ul_mat, off_mat

if Path(npzfile).exists():
    _params   = np.load(npzfile)
    arr_prob  = _params['arr_prob']
    br_dist   = _params['br_dist']
    proc_dist = _params['proc_dist']
    # ul_prob   = _params['ul_prob']
    ul_trans  = _params['ul_trans']
    off_trans = _params['off_trans']
else:
    arr_prob  = 0.05 + 0.05 * np.random.rand(N_AP, N_JOB).astype(np.float32)
    ul_prob   = genUploadingDistribution()
    br_dist   = genDelayDistribution()
    proc_dist = genProcessingDistribution()
    ul_trans, off_trans = genTransitionMatrix(ul_prob)

    np.savez(npzfile, **{
        'arr_prob' : arr_prob,
        'ul_prob'  : ul_prob,
        'proc_dist': proc_dist,
        'ul_trans' : ul_trans,
        'off_trans': off_trans
    })
    pass

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
N_SLT = int(TB/TS)   #
N_CNT = int(3*TB/TS) #maximum uploading time

UL_MIN    = 10*TS
UL_MAX    = N_CNT*TS + 11*TS
UL_RNG    = np.linspace(UL_MIN/TS , UL_MAX/TS, num=N_CNT+1, dtype=np.int32)

PROC_MIN  = 5/TS
PROC_MAX  = 15/TS
PROC_RNG  = np.arange(PROC_MIN, PROC_MAX, dtype=np.int32)
PROC_RNG_L= len(PROC_RNG)
DIM_P     = (LQ+1)*PROC_MAX

npzfile = 'logs/{:05d}.npz'.format(RANDOM_SEED)

@njit
def genProcessingDistribution():
    np.zeros((N_ES, N_JOB, PROC_RNG_L))
    pass

@njit
def genUploadingDistribution():
    pass

if Path(npzfile).exists():
    _params = np.load(npzfile)
else:
    # arr_prob
    # ul_prob
    # ul_dist
    # ul_trans_mat
    # 
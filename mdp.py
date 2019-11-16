
import numpy as np
from params import *
from utility import *
from numba import int32, float32
from numba import njit, prange, jitclass
from itertools import product

@jitclass([ ('ap_stat', int32[:,:,:,:]), ('es_stat', int32[:,:,:]) ])
class State(object):
    def __init__(self):
        self.ap_stat = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.int32)
        self.es_stat = np.zeros((N_ES, N_JOB, 2),           dtype=np.int32)
        pass

    def clone(self, stat):
        self.ap_stat = np.copy(stat.ap_stat)
        self.es_stat = np.copu(stat.es_stat)
        return self
    
    def cost(self):
        return np.sum(self.ap_stat) + np.sum(es_stat[:,:,0])
    pass

@njit
def BaselinePolicy():
    policy = np.zeros((N_AP, N_JOB), dtype=np.int32)
    #TODO:
    return policy

@njit
def AP2Vec(ap_stat):
    ap_vec = np.zeros((N_CNT), dtype=np.float32)
    #TODO:
    return ap_vec

@njit
def ES2Vec(es_stat):
    es_vec = np.zeros((DIM_P), dtype=np.float32)
    #TODO:
    return es_vec

@njit
def TransES(beta, proc_dist):
    mat = np.zeros((DIM_P,DIM_P), dtype=np.float32)
    #TODO:
    return mat

@njit
def evaluate(j, stat, policy):
    val_ap = np.zeros((N_AP, N_ES),        dtype=np.float32)
    val_es = np.zeros((N_ES,),             dtype=np.float32)
    ap_vec = np.zeros((N_AP, N_ES, N_CNT), dtype=np.float32)
    es_vec = np.zeros((N_ES, DIM_P),       dtype=np.float32)
    #TODO:
    return np.sum(val_ap) + np.sum(val_es)

@njit
def optimize(stat):
    policy         = BaselinePolicy()
    val_collection = np.zeros(N_JOB, dtype=np.float32)
    #TODO:
    return policy, val_collection
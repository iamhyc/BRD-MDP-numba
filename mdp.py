
import numpy as np
from params import *
from utility import *
from numba import int32, float64
from numba import njit, prange, jitclass
from itertools import product

ESValVec  = np.repeat(np.arange(LQ+1), repeats=PROC_MAX).astype(np.float64)

@jitclass([ ('ap_stat', int32[:,:,:,:]), ('es_stat', int32[:,:,:]) ])
class State(object):
    def __init__(self):
        self.ap_stat = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.int32)
        self.es_stat = np.zeros((N_ES, N_JOB, 2),           dtype=np.int32)
        pass

    def clone(self, stat):
        self.ap_stat = np.copy(stat.ap_stat)
        self.es_stat = np.copy(stat.es_stat)
        return self
    
    def cost(self):
        return np.sum(self.ap_stat) + np.sum(es_stat[:,:,0])
    pass

@njit
def BaselinePolicy():
    policy = np.zeros((N_AP, N_JOB), dtype=np.int32)
    proc_rng = np.copy(PROC_RNG).astype(np.float64)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            policy[k,j] = (proc_dist[:,j,:] @ proc_rng).argmin()
    return policy

@njit
def AP2Vec(ap_stat, prob):
    assert((ap_stat <= 1).all())
    ap_vec = np.zeros(N_CNT, dtype=np.float64)
    for i in prange(len(ap_vec)):
        ap_vec[i] = ap_stat[i]
    # ap_vec = np.array(ap_stat, dtype=np.float64)
    ap_vec[0] = prob
    return ap_vec

@njit
def ES2Vec(es_stat):
    es_vec = np.zeros((DIM_P), dtype=np.float64)
    _idx   = es_stat[0] * PROC_MAX + es_stat[1]
    es_vec[_idx] = 1
    return es_vec

@njit
def ES2Entry(l,r):
    return l*PROC_MAX + r

@njit
def TransES(beta, proc_dist):
    mat = np.zeros((DIM_P,DIM_P), dtype=np.float64)
    
    #NOTE: fill-in l==0 && r==0
    mat[ES2Entry(0,0), ES2Entry(0,0)] = 1 - beta
    for idx,prob in enumerate(proc_dist):
        mat[ES2Entry(0,0), ES2Entry(0,PROC_RNG[idx])] = prob*beta

    #NOTE: fill-in l!=0 && r==0
    for l in prange(1, LQ+1):
        e = ES2Entry(l,0)
        for idx,prob in enumerate(proc_dist):
            mat[e, ES2Entry(l,   PROC_RNG[idx])] = prob*beta
            mat[e, ES2Entry(l-1, PROC_RNG[idx])] = prob*(1-beta)

    #NOTE: fill-in r!=0
    for l in prange(LQ+1):
        for r in prange(1,PROC_MAX):
            e  = ES2Entry(l,r)
            l2 = LQ if l+1>LQ else (l+1)
            mat[e, ES2Entry(l2, r-1)] += beta
            mat[e, ES2Entry(l,  r-1)] += 1-beta
    
    return mat

@njit
def evaluate(j, _k, systemStat, oldPolicy, nowPolicy):
    (oldStat, nowStat, br_delay) = systemStat
    _delay                       = br_delay[_k]
    val_ap = np.zeros((N_AP, N_ES),        dtype=np.float64)
    val_es = np.zeros((N_ES,),             dtype=np.float64)
    ap_vec = np.zeros((N_AP, N_ES, N_CNT), dtype=np.float64)
    es_vec = np.zeros((N_ES, DIM_P),       dtype=np.float64)

    # generate arrival probability
    old_prob = np.zeros((N_AP, N_ES), dtype=np.float64)
    now_prob = np.zeros((N_AP, N_ES), dtype=np.float64)
    for k in prange(N_AP):
        old_prob[ k, oldPolicy[k] ] = arr_prob[k,j]
        now_prob[ k, nowPolicy[k] ] = arr_prob[k,j]

    # init vector
    for m in prange(N_ES):
        es_vec[m] = ES2Vec(nowStat.es_stat[m,j])
        for k in prange(N_AP):
            ap_vec[k,m] = AP2Vec(nowStat.ap_stat[k,m,j], old_prob[k,m])
    
    # iterate system state to (t+1)
    for n in range(N_SLT):
        for m in prange(N_ES):
            beta = 0.0
            for k in prange(N_AP):
                ap_vec[k,m] = ap_vec[k,m] @ ul_trans[k,m,j]
                beta       += np.sum(ap_vec[k,m] @ off_trans[k,m,j])
                if n==_delay: ap_vec[k,m] = AP2Vec(ap_vec[k,m], now_prob[k,m]) #NOTE: update once is okay!
                pass
            mat       = TransES(beta, proc_dist[m,j])
            es_vec[m] = es_vec[m] @ mat
        pass
    
    # calculate value for AP
    for m in prange(N_ES):
        for k in prange(N_AP):
            mat         = np.copy(ul_trans[k,m,j])
            trans_mat   = np.linalg.matrix_power(mat, N_SLT)
            ident_mat   = np.eye(N_CNT, dtype=np.float64)
            inv_mat     = np.linalg.inv( ident_mat - GAMMA*trans_mat )
            val_ap[k,m] = np.sum( ap_vec[k,m] @ inv_mat )

    # continue iterate system state to (t+3) and collect cost for ES
    for n in range(2*N_SLT):
        for m in prange(N_ES):
            beta = 0.0
            for k in prange(N_AP):
                ap_vec[k,m] =        ap_vec[k,m] @ ul_trans[k,m,j]
                beta       += np.sum(ap_vec[k,m] @ off_trans[k,m,j])
                pass
            mat       = TransES(beta, proc_dist[m,j])
            es_vec[m] = es_vec[m] @ mat
            if n//N_SLT == 0:
                val_es[m] += (es_vec[m] @ ESValVec) * np.power(GAMMA, n//N_SLT) #NOTE: n//N_SLT or n//N_SLT+1 ?
        pass

    # calculate value for ES
    for m in prange(N_ES):
        beta = np.sum(now_prob[:,m]) #FIXME:: TRUE STORY! okay, double-check
        mat  = TransES(beta, proc_dist[m,j])
        trans_mat = np.linalg.matrix_power(mat, N_SLT)
        ident_mat = np.eye(DIM_P, dtype=np.float64)
        inv_mat   = np.linalg.inv( ident_mat - GAMMA*trans_mat )
        val_es[m]+= np.power(GAMMA, 2) * (es_vec[m] @ inv_mat @ ESValVec)

    return np.sum(val_ap) + np.sum(val_es)

@njit
def optimize(stage, systemStat, oldPolicy):
    nowPolicy      = np.copy(oldPolicy)
    val_collection = np.zeros(N_JOB, dtype=np.float64)

    _k = stage // N_AP #NOTE: optimize one AP at one time

    for j in prange(N_JOB):
        x0 = nowPolicy[:, j]
        val_tmp = np.zeros(N_ES, dtype=np.float64)
        for m in prange(N_ES):
            x1         = np.copy(x0)
            x1[_k]     = m
            val_tmp[m] = evaluate(j, _k, systemStat, oldPolicy[:,j], x1)
            pass
        nowPolicy[_k, j] = val_tmp.argmin()
        pass

    for j in prange(N_JOB):
        x1 = nowPolicy[:, j]
        val_collection[j] = evaluate(j, _k, systemStat, oldPolicy[:,j], x1)

    return nowPolicy, val_collection
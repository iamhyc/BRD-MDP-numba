
import numpy as np
from params import *
from utility import *
from numba import int32, int64, float64
from numba import njit, prange, jitclass
from itertools import product

ul_rng     = np.arange(N_CNT, dtype=np.float64)
ESValVec   = np.repeat(np.arange(LQ), repeats=PROC_MAX).astype(np.float64)
PenaltyVec = BETA * np.ones(PROC_MAX, dtype=np.float64)
ESValVec   = np.concatenate((ESValVec, PenaltyVec))

@jitclass([
    ('ap_stat', int32[:,:,:,:]),
    ('es_stat', int32[:,:,:]),
    ('acc_arr', int64),
    ('acc_dep', int64),
    ('acc_cost', int64),
    ('timeslot', int64)
])
class State(object):
    def __init__(self):
        self.ap_stat = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.int32)
        self.es_stat = np.zeros((N_ES, N_JOB, 2),           dtype=np.int32)
        self.acc_arr, self.acc_dep   = 0, 0
        self.acc_cost, self.timeslot = 0, 0
        pass

    def clone(self, stat):
        self.ap_stat = np.copy(stat.ap_stat)
        self.es_stat = np.copy(stat.es_stat)
        self.acc_arr, self.acc_dep  = stat.acc_arr, stat.acc_dep
        self.acc_cost, self.timeslot= stat.acc_cost, stat.timeslot
        return self
    
    def getNumber(self):
        return np.sum(self.ap_stat) + np.sum(self.es_stat[:,:,0])

    def getCost(self):
        _penalty = BETA * np.count_nonzero( self.es_stat[:,:,0]==LQ )
        return _penalty + np.sum(self.ap_stat) + np.sum(self.es_stat[:,:,0])

    def average_JCT(self):
        return self.acc_cost / self.acc_arr
    
    def average_cost(self):
        return self.acc_cost / self.timeslot
    
    def average_throughput(self):
        # return self.acc_dep / self.timeslot
        return self.acc_dep / self.acc_arr

    def getUtility(self):
        return self.acc_dep / self.acc_arr if self.acc_arr!=0 else 0

    def iterate(self, admissions, departures):
        self.timeslot += 1
        self.acc_cost += self.getCost()
        self.acc_arr  += np.sum(admissions)
        self.acc_dep  += np.sum(departures)
        pass

    pass

@njit
def RandomPolicy():
    policy = np.zeros((N_AP, N_JOB), dtype=np.int32)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            policy[k, j] = np.random.randint(N_ES)
    return policy

@njit
def BaselinePolicy():
    policy = np.zeros((N_AP, N_JOB), dtype=np.int32)
    proc_rng = np.copy(PROC_RNG).astype(np.float64)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            # policy[k,j] = (proc_dist[:,j,:] @ proc_rng).argmin()
            policy[k,j] = (ul_prob[k,:,j,:] @ ul_rng + proc_dist[:,j,:] @ proc_rng).argmin()
            # policy[k,j] = (ul_prob[k,:,j,:] @ _tmp).argmin()
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

#FIXME: Matrix under Exponential departure
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

#TODO: improve _k randomness?
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
            beta = np.zeros(N_ES, dtype=np.float64)
            for k in prange(N_AP):
                beta[m]     = np.sum(ap_vec[k,m] @ off_trans[k,m,j])
                ap_vec[k,m] =        ap_vec[k,m] @ ul_trans[k,m,j]                
                if n==_delay: ap_vec[k,m] = AP2Vec(ap_vec[k,m], now_prob[k,m]) #NOTE: update once is okay!
                pass
            mat       = TransES(beta.sum(), proc_dist[m,j])
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
            beta = np.zeros(N_ES, dtype=np.float64)
            for k in prange(N_AP):
                beta[m]     = np.sum(ap_vec[k,m] @ off_trans[k,m,j])
                ap_vec[k,m] =        ap_vec[k,m] @ ul_trans[k,m,j]
                pass
            mat       = TransES(beta.sum(), proc_dist[m,j])
            es_vec[m] = es_vec[m] @ mat
            if n%N_SLT == 0:
                val_es[m] += (es_vec[m] @ ESValVec) * np.power(GAMMA, n//N_SLT)
        pass

    # calculate value for ES
    for m in prange(N_ES):
        _beta = np.sum(now_prob[:,m]) #NOTE:: TRUE STORY! okay, double-check
        mat  = TransES(_beta, proc_dist[m,j])
        trans_mat = np.linalg.matrix_power(mat, N_SLT)
        ident_mat = np.eye(DIM_P, dtype=np.float64)
        inv_mat   = np.linalg.inv( ident_mat - GAMMA*trans_mat )
        val_es[m]+= np.power(GAMMA, 2) * (es_vec[m] @ inv_mat @ ESValVec)

    return np.sum(val_ap) + np.sum(val_es)

@njit
def optimize(stage, systemStat, oldPolicy):
    nowPolicy      = np.copy(oldPolicy)
    val_collection = np.zeros(N_JOB, dtype=np.float64)

    _k = stage % N_AP #NOTE: optimize one AP at one time

    for j in prange(N_JOB):
        val_tmp = np.zeros(N_ES, dtype=np.float64)
        for m in prange(N_ES):
            x1         = np.copy(nowPolicy[:, j])
            x1[_k]     = m
            val_tmp[m] = evaluate(j, _k, systemStat, oldPolicy[:,j], x1)
            pass
        nowPolicy[_k, j]  = val_tmp.argmin()
        val_collection[j] = val_tmp.min()
        pass

    # print(val_collection)

    return nowPolicy, val_collection
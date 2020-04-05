
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
    ('es_stat', int32[:,:]),
    ('acc_arr', int64),
    ('acc_dep', int64),
    ('acc_cost', int64),
    ('timeslot', int64)
])
class State(object):
    def __init__(self):
        self.ap_stat = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.int32)
        self.es_stat = np.zeros((N_ES, N_JOB),           dtype=np.int32)
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
        return np.sum(self.ap_stat) + np.sum(self.es_stat)

    def getCost(self):
        _penalty = BETA * np.count_nonzero( self.es_stat==LQ )
        return _penalty + np.sum(self.ap_stat) + np.sum(self.es_stat)

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
    # proc_rng = np.copy(PROC_RNG).astype(np.float64)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            # policy[k,j] = (proc_mean[:,j]).argmin()
            policy[k,j] = (ul_prob[k,:,j,:] @ ul_rng + proc_mean[:,j]).argmin()
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
    es_vec[es_stat] = 1
    return es_vec

@njit
def TransES(beta, job_mean):
    mat = np.zeros((DIM_P,DIM_P), dtype=np.float64)
    
    # fill-in l1==0
    mat[0, 0] = 1-beta
    mat[0, 1] = beta
    # fill-in l1 < LQ
    for l1 in prange(1, DIM_P-1):
        mat[l1, l1-1] = (1/job_mean) * (1-beta)
        mat[l1, l1]   = (1-1/job_mean)*(1-beta) + (1/job_mean)*beta
        mat[l1, l1+1] = (1-1/job_mean)*beta
    # fill-in l1==LQ
    mat[LQ, LQ-1] = 1/job_mean
    mat[LQ, LQ]   = 1-1/job_mean
    
    return mat

@njit(parallel=True)
def evaluate(j, _k, systemStat, oldPolicy, nowPolicy):
    (oldStat, nowStat, br_delay) = systemStat
    _delay                       = br_delay[_k]
    can_set = np.where( bi_map[_k]==1 )[0]
    N_CAN   = len(can_set)
    val_ap  = np.zeros((N_AP, N_CAN),        dtype=np.float64) #here not restrict conflict set size
    val_es  = np.zeros((N_CAN,),             dtype=np.float64)
    ap_vec  = np.zeros((N_AP, N_CAN, N_CNT), dtype=np.float64)
    es_vec  = np.zeros((N_CAN, DIM_P),       dtype=np.float64)

    # generate arrival probability
    old_prob = np.zeros((N_AP, N_ES), dtype=np.float64)
    now_prob = np.zeros((N_AP, N_ES), dtype=np.float64)
    for k in prange(N_AP):
        old_prob[ k, oldPolicy[k] ] = arr_prob[k,j]
        now_prob[ k, nowPolicy[k] ] = arr_prob[k,j]

    # init vector
    for m in prange(N_ES):
        if bi_map[_k,m]:
            es_vec[m] = ES2Vec(nowStat.es_stat[m,j])                         #only (_k)'s candidate set
            for k in prange(N_AP):
                ap_vec[k,m] = bi_map[k,m] * AP2Vec(nowStat.ap_stat[k,m,j], old_prob[k,m])   #only (m)'s conflict set
        pass
    
    # iterate system state to (t+1)
    for n in range(N_SLT):
        for m in prange(N_ES):
            beta = np.zeros(N_ES, dtype=np.float64)
            for k in prange(N_AP):
                beta[m]     = np.sum(ap_vec[k,m] @ off_trans[k,m,j])
                ap_vec[k,m] =        ap_vec[k,m] @ ul_trans[k,m,j]                
                if n==_delay: ap_vec[k,m] = AP2Vec(ap_vec[k,m], now_prob[k,m]) #NOTE: update once is okay!
                pass
            mat       = TransES(beta.sum(), proc_mean[m,j])
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
            mat       = TransES(beta.sum(), proc_mean[m,j])
            es_vec[m] = es_vec[m] @ mat
            if n%N_SLT == 0:
                val_es[m] += (es_vec[m] @ ESValVec) * np.power(GAMMA, n//N_SLT)
        pass

    # calculate value for ES
    for m in prange(N_ES):
        _beta = np.sum(now_prob[:,m]) #NOTE:: TRUE STORY!
        mat  = TransES(_beta, proc_mean[m,j])
        trans_mat = np.linalg.matrix_power(mat, N_SLT)
        ident_mat = np.eye(DIM_P, dtype=np.float64)
        inv_mat   = np.linalg.inv( ident_mat - GAMMA*trans_mat )
        val_es[m]+= np.power(GAMMA, 2) * (es_vec[m] @ inv_mat @ ESValVec)

    return np.sum(val_ap) + np.sum(val_es)

@njit(parallel=True)
def optimize(stage, systemStat, oldPolicy):
    nowPolicy      = np.copy(oldPolicy)
    val_collection = np.zeros(N_JOB, dtype=np.float64)

    _n = stage % N_SET #NOTE: optimize multiple APs at one time

    for k in prange(subset_numba[_n]):                          #iterate over current subset
        if subset_numba[_n,k]:                                  #(cont.):
            for j in prange(N_JOB):                             #   iterate the job space:
                val_tmp = np.zeros(N_ES, dtype=np.float64)      #   |
                for m in prange(N_ES):                          #   |   iterate its candidate set
                    if bi_map[k,m]:                             #   |   (cont.):
                        x1         = np.copy(nowPolicy[:,j])    #   |   |
                        x1[k]      = m                          #   |   |
                        val_tmp[m] = evaluate(j, k, systemStat, oldPolicy[:,j], x1)
                    pass                                        #   |   end
                nowPolicy[k, j]  = val_tmp.argmin()             #   |
                val_collection[j] = val_tmp.min()               #   |
                pass                                            #   end
        pass                                                    #end

    # print(val_collection)

    return nowPolicy, val_collection

import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib.pyplot as plt

def NextState(arrival_ap, systemStat, oldPolicy, nowPolicy):
    (oldStat, nowStat, br_delay) = systemStat 
    lastStat  = State().clone(nowStat)
    nextStat  = State().clone(lastStat)

    # update intermediate state with arrivals in each time slot 
    for n in range(N_SLT):
        #NOTE: allocate arrival jobs on APs
        for j in range(N_JOB):
            for k in range(N_AP):
                if callable(oldPolicy) and callable(nowPolicy):
                    _m = oldPolicy(oldStat, k, j) if n<br_delay[k] else nowPolicy(nowStat, k, j)
                else:
                    _m = oldPolicy[k,j]           if n<br_delay[k] else nowPolicy[k,j]
                nextStat[k, _m, j, 0] = arrival_ap[n, k, j]
        
        #NOTE: count uploading & offloading jobs
        off_number = np.zeros((N_ES, N_JOB), dtype=np.int32)
        for xi in range(N_CNT):
            for j in range(N_JOB):
                for m in range(N_ES):
                    for k in range(N_AP):
                        toss_ul = toss(ul_prob[k,m,j,xi]) #NOTE: hidden_assert(if xi==N_CNT-1 then: ul_prob==1)
                        if toss_ul:
                            off_number[m,j]             += lastStat.ap_stat[k,m,j,xi]
                        else:
                            nextStat.ap_stat[k,m,j,xi+1] = lastStat.ap_stat[k,m,j,xi]

        #NOTE: process jobs on ES
        for j in range(N_JOB):
            for m in range(N_ES):
                nextStat.es_stat[m,j,0] += off_number[m,j]
                nextStat.es_stat[m,j,1] -= 1

                if nextStat.es_stat[m,j,0] > LQ:            # CLIP [0, LQ]
                    nextStat.es_stat[m,j,0] = LQ            #
                if nextStat.es_stat[m,j,1] <= 0:            # if first job finished:
                    if nextStat.es_stat[m,j,0] > 0:         #     if has_next_job:
                        nextStat.es_stat[m,j,0] -= 1        #         next job joins processing
                        nextStat.es_stat[m,j,1]  = PROC_RNG[ multoss(proc_dist[m,j]) ]
                    else:                                   #     else:
                        nextStat.es_stat[m,j,1]  = 0        #         clip the remaining time
                else:                                       # else:
                    pass                                    #     do nothing.

        #NOTE: update the iteration backup
        lastStat = nextStat
        nextStat = State().clone(lastStat)
        pass

    return nextStat

def main():
    pathlib.Path('logs').mkdir(exist_ok=True)
    pathlib.Path('figures').mkdir(exist_ok=True)

    stage = 0
    oldStat,   nowStat   = State(),          State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    
    while stage < STAGE:
        #NOTE: toss job arrival for APs in each time slot
        arrival_ap = np.zeros((N_SLT, N_AP, N_JOB), dtype=np.int32)
        for n in range(N_SLT):
            for j in range(N_JOB):
                for k in range(N_AP):
                    arrival_ap[n,k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

        #NOTE: toss broadcast delay for each AP
        br_delay = np.zeros((N_AP), dtype=np.int32)
        for k in range(N_AP):
            br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]

        #NOTE: optimize and update the backup
        systemStat     = (oldStat, nowStat, br_delay)
        oldPolicy      = nowPolicy
        nowPolicy, val = optimize(stage, systemStat, oldPolicy)
        oldStat        = nowStat
        nowStat        = NextState(arrival_ap, systemStat, oldPolicy, nowPolicy)
        
        stage += 1
        pass
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        pass
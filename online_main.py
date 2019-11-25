import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib.pyplot as plt
from termcolor import cprint

ul_rng    = np.arange(N_CNT, dtype=np.float64)

@njit
def ABaselinPolicy(stat, k, j):
    return BaselinePolicy()[k,j]

@njit
def ARandomPolicy(stat, k, j):
    return np.random.randint(N_ES)

@njit
def AQueueFirstPolicy(stat, k, j):
    return (stat.es_stat[:,j,0]).argmin()

@njit
def ASelfishPolicy(stat, k, j):
    proc_rng  = np.copy(PROC_RNG).astype(np.float64)
    eval_cost = ul_prob[k,:,j,:] @ ul_rng + proc_dist[:,j,:] @ proc_rng
    # eval_cost = ul_prob[k,:,j,:] @ ul_rng + (stat.es_stat[:,j,0]+1)*(proc_dist[:,j,:] @ proc_rng)
    return eval_cost.argmin()

def NextState(arrival_ap, systemStat, oldPolicy, nowPolicy):
    (oldStat, nowStat, br_delay) = systemStat 
    lastStat  = State().clone(nowStat)
    nextStat  = State().clone(lastStat)

    # print(arrival_ap)

    # update intermediate state with arrivals in each time slot 
    for n in range(N_SLT):
        nextStat.ap_stat = np.zeros((N_AP,N_ES,N_JOB,N_CNT), dtype=np.int32)
        #NOTE: allocate arrival jobs on APs
        for j in range(N_JOB):
            for k in range(N_AP):
                if callable(oldPolicy) and callable(nowPolicy):
                    _m = oldPolicy(oldStat, k, j) if n<br_delay[k] else nowPolicy(nowStat, k, j)
                else:
                    _m = oldPolicy[k,j]           if n<br_delay[k] else nowPolicy[k,j]
                nextStat.ap_stat[k, _m, j, 0] = arrival_ap[n, k, j]
        
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
            pass

        #NOTE: update the iteration backup
        # print(np.sum(nextStat.ap_stat))
        # print(nextStat.es_stat[:,:,0])
        lastStat = nextStat
        nextStat = State().clone(lastStat)
        pass

    return nextStat

def main():
    pathlib.Path('./traces-{:05d}'.format(RANDOM_SEED)).mkdir(exist_ok=True)
    
    stage = 0
    oldStat,   nowStat   = State(),          State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    #-----------------------------------------------------------
    bs_oldStat, bs_nowStat = State(), State()
    sf_oldStat, sf_nowStat = State(), State()
    qf_oldStat, qf_nowStat = State(), State()
    rd_oldStat, rd_nowStat = State(), State()
    #-----------------------------------------------------------

    print('Baseline Policy\n{}'.format(nowPolicy))

    while stage < STAGE:
        with Timer(output=True):
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
            #----------------------------------------------------------------
            systemStat             = (bs_oldStat, bs_nowStat, br_delay)
            bs_oldStat, bs_nowStat = bs_nowStat, NextState(arrival_ap, systemStat, ABaselinPolicy, ABaselinPolicy)
            systemStat             = (sf_oldStat, sf_nowStat, br_delay)
            sf_oldStat, sf_nowStat = sf_nowStat, NextState(arrival_ap, systemStat, ASelfishPolicy, ASelfishPolicy)
            systemStat             = (qf_oldStat, qf_nowStat, br_delay)
            qf_oldStat, qf_nowStat = qf_nowStat, NextState(arrival_ap, systemStat, AQueueFirstPolicy, AQueueFirstPolicy)
            systemStat             = (rd_oldStat, rd_nowStat, br_delay)
            rd_oldStat, rd_nowStat = rd_nowStat, NextState(arrival_ap, systemStat, ARandomPolicy, ARandomPolicy)
            #----------------------------------------------------------------

            cprint('Stage-{} Delta Policy'.format(stage), 'red')
            print(nowPolicy - oldPolicy)
            cprint('ES State:', 'green')
            print(nowStat.es_stat[:,:,0])
            
            stage += 1
        pass

        #---------------------------------------------------------------------
        plt.plot([stage, stage+1], [oldStat.cost(), nowStat.cost()], '-ro')
        plt.plot([stage, stage+1], [bs_oldStat.cost(), bs_nowStat.cost()], '-ko')
        plt.plot([stage, stage+1], [sf_oldStat.cost(), sf_nowStat.cost()], '-bo')
        plt.plot([stage, stage+1], [qf_oldStat.cost(), qf_nowStat.cost()], '-go')
        plt.plot([stage, stage+1], [rd_oldStat.cost(), rd_nowStat.cost()], '-co')
        plt.legend(['MDP Policy', 'Baseline Policy', 'Selfish Policy', 'SQF Policy', 'Random Policy'])
        #---------------------------------------------------------------------
        plt.pause(0.05)

        trace_file = 'traces-{:05d}/{:04d}.npz'.format(RANDOM_SEED, stage)
        np.savez(trace_file, **{
            'mdp_ap_stat': nowStat.ap_stat,
            'mdp_es_stat': nowStat.es_stat,
            'mdp_value'  : val
        })
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        pass
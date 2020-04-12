#!/usr/bin/env python3
import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib
import matplotlib.pyplot as plt
from termcolor import cprint

ul_rng    = np.arange(N_CNT, dtype=np.float64)

PLOT_FLAG = False
if PLOT_FLAG: matplotlib.use("Qt5agg")

@njit
def ARandomPolicy(stat, k, j):
    return np.random.randint(N_ES)

@njit
def ASelfishPolicy(stat, k, j):
    # proc_rng  = np.copy(PROC_RNG).astype(np.float64)
    eval_cost = ul_prob[k,:,j,:] @ ul_rng + proc_mean[:,j]
    return eval_cost.argmin()

@njit
def AQueueAwarePolicy(stat, k, j):
    # proc_rng  = np.copy(PROC_RNG).astype(np.float64)
    eval_cost = ul_prob[k,:,j,:] @ ul_rng + (stat.es_stat[:,j]+1)* proc_mean[:,j]
    return eval_cost.argmin()
    # return (stat.es_stat[:,j]).argmin()

@njit
def ARealBenchmark(stat, k, j):
    eval_cost = [] #TODO: add one benchmark
    return eval_cost.argmin()

def NextState(arrivals, systemStat, oldPolicy, nowPolicy):
    (oldStat, nowStat, br_delay) = systemStat 
    lastStat  = State().clone(nowStat)
    nextStat  = State().clone(lastStat)

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
                nextStat.ap_stat[k, _m, j, 0] = arrivals[n, k, j]
        
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
        departures = np.zeros((N_ES, N_JOB), dtype=np.int32)
        for j in range(N_JOB):
            for m in range(N_ES):
                nextStat.es_stat[m,j] += off_number[m,j]

                if nextStat.es_stat[m,j] > LQ:                  # CLIP [0, LQ]
                    nextStat.es_stat[m,j] = LQ                  #
                _completed = toss(1/proc_mean[m,j])             # toss for the first job, if exist
                if (nextStat.es_stat[m,j]>0) and _completed:    # if first_job_completed:
                    departures[m,j]       += 1                  #       record departure;
                    nextStat.es_stat[m,j] -= 1                  #       job departure;
                else:                                           # else:
                    pass                                        #       do nothing.
                pass
            pass

        #NOTE: update the iteration backup
        # print(np.sum(departures))
        nextStat.iterate(off_number, departures) #update the accumulation
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
    SF_oldStat, SF_nowStat = State(), State()
    QA_oldStat, QA_nowStat = State(), State()
    RD_oldStat, RD_nowStat = State(), State()
    #-----------------------------------------------------------

    print('Baseline Policy\n{}'.format(nowPolicy))

    if PLOT_FLAG: plt.ion()
    while stage < STAGE:
        with Timer(output=True):
            #NOTE: toss job arrival for APs in each time slot
            arrivals = np.zeros((N_SLT, N_AP, N_JOB), dtype=np.int32)
            for n in range(N_SLT):
                for j in range(N_JOB):
                    for k in range(N_AP):
                        arrivals[n,k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

            #NOTE: toss broadcast delay for each AP
            br_delay = np.zeros((N_AP), dtype=np.int32)
            for k in range(N_AP):
                br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]

            #NOTE: optimize and update the backup
            systemStat     = (oldStat, nowStat, br_delay)
            oldPolicy      = nowPolicy
            nowPolicy, val = optimize(stage, systemStat, oldPolicy)
            oldStat        = nowStat
            nowStat        = NextState(arrivals, systemStat, oldPolicy, nowPolicy)
            #----------------------------------------------------------------
            systemStat             = (SF_oldStat, SF_nowStat, br_delay)
            SF_oldStat, SF_nowStat = SF_nowStat, NextState(arrivals, systemStat, ASelfishPolicy, ASelfishPolicy)
            systemStat             = (QA_oldStat, QA_nowStat, br_delay)
            QA_oldStat, QA_nowStat = QA_nowStat, NextState(arrivals, systemStat, AQueueAwarePolicy, AQueueAwarePolicy)
            systemStat             = (RD_oldStat, RD_nowStat, br_delay)
            RD_oldStat, RD_nowStat = RD_nowStat, NextState(arrivals, systemStat, ARandomPolicy, ARandomPolicy)
            #----------------------------------------------------------------

            cprint('Stage-{} Delta Policy'.format(stage), 'red')
            print(nowPolicy - oldPolicy)
            cprint('ES State:', 'green')
            print(nowStat.es_stat)
            
            stage += 1
        pass

        if PLOT_FLAG:
            #---------------------------------------------------------------------
            plt.plot([stage, stage+1], [oldStat.getNumber(), nowStat.getNumber()], '-ro')
            plt.plot([stage, stage+1], [SF_oldStat.getNumber(), SF_nowStat.getNumber()], '-bo')
            plt.plot([stage, stage+1], [QA_oldStat.getNumber(), QA_nowStat.getNumber()], '-go')
            plt.plot([stage, stage+1], [RD_oldStat.getNumber(), RD_nowStat.getNumber()], '-co')
            plt.legend(['MDP Policy', 'Selfish Policy', 'SQF Policy', 'Random Policy'])
            #---------------------------------------------------------------------
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.3)

        trace_file = 'traces-{:05d}/{:04d}.npz'.format(RANDOM_SEED, stage)
        np.savez(trace_file, **{
            'MDP_value'   : val,
            'MDP_ap_stat' : nowStat.ap_stat,
            'MDP_es_stat' : nowStat.es_stat,
            "Selfish_ap_stat": SF_nowStat.ap_stat,
            "Selfish_es_stat": SF_nowStat.es_stat,
            "QAware_ap_stat" : QA_nowStat.ap_stat,
            "QAware_es_stat" : QA_nowStat.es_stat,
            "Random_ap_stat" : RD_nowStat.ap_stat,
            "Random_es_Stat" : RD_nowStat.es_stat
        })

        print('Cost:', nowStat.getCost(), SF_nowStat.getCost(), QA_nowStat.getCost(), RD_nowStat.getCost())
        print('Burden:', nowStat.getUtility(), SF_nowStat.getUtility(), QA_nowStat.getUtility(), RD_nowStat.getUtility())
        pass

    #save summary file
    summary_file = 'traces-{:05d}/summary'.format(RANDOM_SEED)
    np.savez(summary_file, **{
        'MDP_average_cost'    : nowStat.average_cost(),
        'Selfish_average_cost': SF_nowStat.average_cost(),
        'QAware_average_cost' : QA_nowStat.average_cost(),
        'Random_average_cost' : RD_nowStat.average_cost(),
        'MDP_average_JCT'    : nowStat.average_JCT(),
        'Selfish_average_JCT': SF_nowStat.average_JCT(),
        'QAware_average_JCT' : QA_nowStat.average_JCT(),
        'Random_average_JCT' : RD_nowStat.average_JCT(),
        'MDP_average_throughput'    : nowStat.average_throughput(),
        'Selfish_average_throughput': SF_nowStat.average_throughput(),
        'QAware_average_throughput' : QA_nowStat.average_throughput(),
        'Random_average_throughput' : RD_nowStat.average_throughput()
    })

    # print(nowStat.average_cost(), SF_nowStat.average_cost(), QA_nowStat.average_cost(), RD_nowStat.average_cost())
    print(nowStat.getUtility(), SF_nowStat.getUtility(), QA_nowStat.getUtility(), RD_nowStat.getUtility())
    # if PLOT_FLAG: plt.show()
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        pass

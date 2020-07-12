#!/usr/bin/env python3
import os, pathlib
import argparse
import numpy as np
from mdp import *
from params import *
from utility import *
import matplotlib
import matplotlib.pyplot as plt
from termcolor import cprint

RECORD_PREFIX = '{:05d}'.format(RANDOM_SEED)

@njit
def ARandomPolicy(stat, k, j):
    _can_set = np.where( bi_map[k]==1 )[0]
    return np.random.choice(_can_set)

@njit
def ASelfishPolicy(stat, k, j):
    eval_cost     = ul_prob[k,:,j,:] @ ul_rng + proc_mean[:,j]
    eval_cost    -= int(1E9) * bi_map[k]
    return_choice = eval_cost.argmin()
    assert( bi_map[k,return_choice]==1 ) #NOTE: restrict for candidate set
    return return_choice

@njit
def AQueueAwarePolicy(stat, k, j):
    eval_cost     = ul_prob[k,:,j,:] @ ul_rng + (stat.es_stat[:,j]+1)* proc_mean[:,j]
    eval_cost    -= int(1E9) * bi_map[k]
    return_choice = eval_cost.argmin() #(stat.es_stat[:,j]).argmin()
    assert( bi_map[k,return_choice]==1 ) #NOTE: restrict for candidate set
    return return_choice

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
                assert( bi_map[k,_m]==1 )
                nextStat.ap_stat[k, _m, j, 0] = arrivals[n, k, j]
        
        #NOTE: count uploading & offloading jobs (AP to ES)
        off_number = np.zeros((N_ES, N_JOB), dtype=np.int32)
        for xi in range(N_CNT):
            for j in range(N_JOB):
                for m in range(N_ES):
                    for k in range(N_AP):
                        toss_ul = toss(ul_prob[k,m,j,xi]) #NOTE: hidden_assert(ul_prob==1, when xi==N_CNT-1)
                        if toss_ul:
                            off_number[m,j]             += lastStat.ap_stat[k,m,j,xi]
                        else:
                            nextStat.ap_stat[k,m,j,xi+1] = lastStat.ap_stat[k,m,j,xi]

        #NOTE: process jobs on ES
        departures = np.zeros((N_ES, N_JOB), dtype=np.int32)
        nextStat.es_stat += off_number
        nextStat.es_stat  = np.clip(nextStat.es_stat, 0, LQ)
        for j in range(N_JOB):
            for m in range(N_ES):
                if nextStat.es_stat[m,j]>0:
                    completed_num            = 1 if toss(1/proc_mean[m,j]) else 0
                    nextStat.es_stat[m,j]   -= completed_num
                    departures[m,j]         += completed_num
                else:
                    nextStat.es_stat[m,j]    = 0
                pass
            pass

        #NOTE: update the iteration backup
        # print(np.sum(departures))
        nextStat.iterate(off_number, departures) #update the accumulation
        lastStat = nextStat
        nextStat = State().clone(lastStat)
        pass

    return nextStat

def main(args):
    record_mark = '{prefix}-{postfix}'.format(prefix=RECORD_PREFIX, postfix=args.postfix)
    if args.plot_flag:
        matplotlib.use("Qt5agg")
        plt.ion()
    print(record_mark)
    logger = getLogger(record_mark)
    pathlib.Path('./traces-{}'.format(record_mark)).mkdir(exist_ok=True)
    
    stage = 0
    oldStat,   nowStat   = State(),          State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    #-----------------------------------------------------------
    SF_oldStat, SF_nowStat = State(), State()
    QA_oldStat, QA_nowStat = State(), State()
    RD_oldStat, RD_nowStat = State(), State()
    #-----------------------------------------------------------

    print('Baseline Policy\n{}'.format(nowPolicy))
    
    while stage < STAGE:
        with Timer(output=True):
            #NOTE: load trace from pre-defined trace folder (looped)
            arrivals = loadArrivalTrace(stage, arr_trace) #toss(arr_prob[k,j])
            assert( np.any(arrivals==1) )

            #NOTE: toss broadcast delay for each AP
            br_delay = np.zeros((N_AP), dtype=np.int32)
            for k in range(N_AP):
                br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]

            #NOTE: optimize and update the backup
            systemStat     = (oldStat, nowStat, br_delay)
            oldPolicy      = nowPolicy
            if args.serial_flag:
                nowPolicy, val = serial_optimize(stage, systemStat, oldPolicy)
            else:
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

            #NOTE: console output with State and Policy deviation
            cprint('Stage-{} Delta Policy'.format(stage), 'red')
            print(nowPolicy - oldPolicy)
            cprint('ES State:', 'green')
            print(nowStat.es_stat)
            
            stage += 1
        pass

        if args.plot_flag:
            #---------------------------------------------------------------------
            plt.plot([stage, stage+1], [oldStat.getCost(), nowStat.getCost()], '-ro')
            plt.plot([stage, stage+1], [SF_oldStat.getCost(), SF_nowStat.getCost()], '-bo')
            plt.plot([stage, stage+1], [QA_oldStat.getCost(), QA_nowStat.getCost()], '-go')
            plt.plot([stage, stage+1], [RD_oldStat.getCost(), RD_nowStat.getCost()], '-co')
            plt.legend(['MDP Policy', 'Selfish Policy', 'SQF Policy', 'Random Policy'])
            #---------------------------------------------------------------------
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.3)
        else:
            m1, m2, m3, m4 = nowStat.getNumber(),SF_nowStat.getNumber(),QA_nowStat.getNumber(),RD_nowStat.getNumber()
            logger.debug('Stage-{}: MDP \t Selfish \t Queue \t Random'.format(stage))
            logger.debug('\t\t\t%3d \t %2d \t\t %2d \t %2d'%(m1, m2-m1, m3-m1, m4-m1))
            pass

        trace_file = 'traces-{}/{:04d}.npz'.format(record_mark, stage)
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

        # logger.debug( 'Cost:{}, {}, {}, {}'.format(nowStat.getCost(), SF_nowStat.getCost(), QA_nowStat.getCost(), RD_nowStat.getCost()) )
        # logger.debug( 'Burden:{}, {}, {}, {}'.format(nowStat.getUtility(), SF_nowStat.getUtility(), QA_nowStat.getUtility(), RD_nowStat.getUtility()) )
        pass
    
    #NOTE: blame remaining jobs to throughput
    empty_admissions = np.zeros((N_AP, N_ES, N_JOB, N_CNT), dtype=np.int32)
    nowStat.iterate(empty_admissions, nowStat.es_stat)
    SF_nowStat.iterate(empty_admissions, SF_nowStat.es_stat)
    QA_nowStat.iterate(empty_admissions, QA_nowStat.es_stat)
    RD_nowStat.iterate(empty_admissions, RD_nowStat.es_stat)
    
    #save summary file
    summary_file = 'traces-{}/summary'.format(record_mark)
    np.savez(summary_file, **{
        'MDP_average_cost'    : nowStat.average_cost(),
        'Selfish_average_cost': SF_nowStat.average_cost(),
        'QAware_average_cost' : QA_nowStat.average_cost(),
        'Random_average_cost' : RD_nowStat.average_cost(),

        'MDP_average_JCT'    : nowStat.average_JCT(),
        'Selfish_average_JCT': SF_nowStat.average_JCT(),
        'QAware_average_JCT' : QA_nowStat.average_JCT(),
        'Random_average_JCT' : RD_nowStat.average_JCT(),

        '_MDP_average_JCT'    : nowStat._average_JCT(),
        '_Selfish_average_JCT': SF_nowStat._average_JCT(),
        '_QAware_average_JCT' : QA_nowStat._average_JCT(),
        '_Random_average_JCT' : RD_nowStat._average_JCT(),

        'MDP_average_throughput'    : nowStat.average_throughput(),
        'Selfish_average_throughput': SF_nowStat.average_throughput(),
        'QAware_average_throughput' : QA_nowStat.average_throughput(),
        'Random_average_throughput' : RD_nowStat.average_throughput()
    })
    try:
        os.rename(summary_file+'.npz', summary_file) #remove ".npz" for summary file
    except Exception as e:
        print('No summary file found.')

    logger.debug('Average Cost: {}, {}, {}, {}'.format( nowStat.average_cost(), SF_nowStat.average_cost(), QA_nowStat.average_cost(), RD_nowStat.average_cost() ))
    logger.debug('Utility: {}, {}, {}, {}'.format( nowStat.getUtility(), SF_nowStat.getUtility(), QA_nowStat.getUtility(), RD_nowStat.getUtility() ))
    if args.plot_flag: plt.show()
    pass

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Main entry to BRD MDP simulation.')
        parser.add_argument('--serial-optimize', dest='serial_flag', action='store_true', default=False,
            help='Use serial optimization in MDP method.')
        parser.add_argument('--plot', dest='plot_flag', action='store_true', default=False,
            help='Plot figure (with Qt5) while running simulation.')
        parser.add_argument('--postfix', dest='postfix', type=str, default='',
            help='specify postfix for record path/files.')
        args = parser.parse_args()

        main(args)
    except Exception as e:
        raise e
    finally:
        pass

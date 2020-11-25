#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
from sys import prefix
import numpy as np
from mdp import *
from params import *
from utility import *
import matplotlib
import matplotlib.pyplot as plt
from termcolor import cprint
from itertools import product

RECORD_PREFIX = '{:05d}'.format(RANDOM_SEED)
NONE_POLICY   = np.zeros((N_AP, N_JOB), dtype=np.int32)

@njit()
def ARandomPolicy(stat, k, j):
    _can_set = np.where( bi_map[k]==1 )[0]
    if k==e_k and j==e_j:
        return e_m
    else:
        return np.random.choice(_can_set)

@njit()
def ASelfishPolicy(stat, k, j):
    eval_cost     = ul_prob[k,:,j,:] @ ul_rng + proc_mean[:,j]
    eval_cost    -= int(1E9) * bi_map[k]
    return_choice = eval_cost.argmin()
    assert( bi_map[k,return_choice]==1 ) #NOTE: restrict for candidate set
    return return_choice

@njit()
def AQueueAwarePolicy(stat, k, j):
    eval_cost     = ul_prob[k,:,j,:] @ ul_rng + (stat.es_stat[:,j]+1)* proc_mean[:,j]
    eval_cost    -= int(1E9) * bi_map[k]
    return_choice = eval_cost.argmin() #(stat.es_stat[:,j]).argmin()
    assert( bi_map[k,return_choice]==1 ) #NOTE: restrict for candidate set
    return return_choice

# @Timer.timeit
@njit(parallel=False)
def NextState(arrivals, systemStat, oldPolicy, nowPolicy, oldPolicyFn, nowPolicyFn):
    (oldStat, nowStat, br_delay) = systemStat
    lastStat  = State().clone(nowStat)
    nextStat  = State().clone(lastStat)

    # update intermediate state with arrivals in each time slot 
    for n in range(N_SLT):
        nextStat.ap_stat = np.zeros((N_AP,N_ES,N_JOB,N_CNT), dtype=np.int32)
        #NOTE: allocate arrival jobs on APs
        for j in range(N_JOB):
            for k in range(N_AP):
                if oldPolicyFn is not None: #callable(oldPolicy) and callable(nowPolicy):
                    _m = oldPolicyFn(oldStat, k, j) if n<br_delay[k] else nowPolicyFn(nowStat, k, j)
                else:
                    _m = oldPolicy[k,j]             if n<br_delay[k] else nowPolicy[k,j]
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
        # nextStat.es_stat = np.clip(nextStat.es_stat, 0, LQ)
        for j in range(N_JOB):
            for m in range(N_ES):
                if nextStat.es_stat[m,j]>LQ:
                    nextStat.es_stat[m,j]=LQ
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

def main_param_fitting(args):
    global e_k, e_m, e_j
    #reference: https://matplotlib.org/gallery/api/two_scales.html
    matplotlib.use("Qt5agg")
    plt.ion()
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.set_xlabel('Index of Broadcast Intervals')
    ax1.set_ylabel('probability')
    # ax1.set_xlim(0, 99)
    ax1.set_ylim([-0.00, 0.10])
    ax2 = ax1.twinx()
    ax2.set_ylabel('time slots')
    ax2.set_ylim([0, 75])
    #
    stage = 0
    e_lambda0 = np.zeros((N_AP, N_JOB), dtype=np.float32)
    t_c       = np.zeros((N_ES, N_JOB), dtype=np.float32)
    e_c0      = np.zeros((N_ES, N_JOB), dtype=np.float32)
    e_u0      = np.zeros((N_AP,N_ES,N_JOB,N_CNT), dtype=np.float32) #not probability but counter
    e_k       = 3
    e_m       = 1 #np.where( bi_map[e_k]==1 )[0]
    e_j       = 0 #random.randint(0, N_JOB); print('j =',e_j) #0
    oldStat, nowStat = State(), State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    #
    while stage < 100:
        # state simulate with MDP_POLICY
        arrivals = loadArrivalTrace(stage)
        br_delay = np.zeros((N_AP), dtype=np.int32) #no delay needed
        systemStat     = (oldStat, nowStat, br_delay)
        oldPolicy      = nowPolicy
        nowPolicy, _   = optimize(stage, systemStat, oldPolicy)
        #----------------------------------------------------------------
        e_lambda = e_lambda0.copy(); e_u = e_u0.copy(); e_c = e_c0.copy()
        lastStat  = State().clone(nowStat)
        nextStat  = State().clone(lastStat)
        for n in range(N_SLT):
            # 1. estimation of arrival probability
            # print(e_lambda)
            t = stage * N_SLT + n + 1
            e_lambda = (t-1)/t * e_lambda + (1/t) * arrivals[n]
            nextStat.ap_stat = np.zeros((N_AP,N_ES,N_JOB,N_CNT), dtype=np.int32)
            for j in range(N_JOB):
                for k in range(N_AP):
                    _m = ARandomPolicy(oldStat, k, j) if n<br_delay[k] else ARandomPolicy(nowStat, k, j)
                    # _m = oldPolicy[k,j] if n<br_delay[k] else nowPolicy[k,j]
                    assert( bi_map[k,_m]==1 )
                    nextStat.ap_stat[k, _m, j, 0] = arrivals[n, k, j]
            # 2. estimation of mean uploading time
            # print(e_u)
            off_number = np.zeros((N_ES, N_JOB), dtype=np.int32)
            for xi in range(N_CNT):
                for j in range(N_JOB):
                    for m in range(N_ES):
                        for k in range(N_AP):
                            toss_ul = toss(ul_prob[k,m,j,xi])
                            if toss_ul:
                                off_number[m,j]             += lastStat.ap_stat[k,m,j,xi]
                                e_u[k,m,j,xi]               += lastStat.ap_stat[k,m,j,xi]
                            else:
                                nextStat.ap_stat[k,m,j,xi+1] = lastStat.ap_stat[k,m,j,xi]
            nextStat.es_stat += off_number
            # 3. estimation of mean computation time
            # print(np.abs(e_c - proc_mean)); print()
            for j in range(N_JOB):
                for m in range(N_ES):
                    if nextStat.es_stat[m,j]>LQ:
                        nextStat.es_stat[m,j]=LQ
                    if nextStat.es_stat[m,j]>0:
                        _success = toss(1/proc_mean[m,j])
                        if _success:
                            t_c[m,j] += 1
                            e_c[m,j] = (t_c[m,j]-1)/t_c[m,j] * e_c[m,j] + (1/t_c[m,j]) * np.random.geometric(1/proc_mean[m,j])
                        completed_num            = 1 if _success else 0
                        nextStat.es_stat[m,j]   -= completed_num 
                    else:
                        nextStat.es_stat[m,j]    = 0
                    pass
            lastStat = nextStat
            nextStat = State().clone(lastStat)
            pass
        #to next interval
        oldStat, nowStat = nowStat, nextStat
        #---------------------------------------------------------------------
        e_u_mean  = np.sum( np.arange(N_CNT) * normalize(e_u[e_k,e_m,e_j]) )
        e_u0_mean = np.sum( np.arange(N_CNT) * normalize(e_u0[e_k,e_m,e_j]) )
        ul_mean   = np.sum( np.arange(N_CNT) * ul_dist[e_k,e_m,e_j] ) #print(e_u_mean, ul_mean)
        # print(e_u[e_k,e_m,e_j], '\n')
        #plot estimated value
        _ln1e = ax1.plot((stage,stage+1), (e_lambda0[e_k,e_j],e_lambda[e_k,e_j]), '-r.')
        _ln2e = ax2.plot((stage,stage+1), (e_u0_mean, e_u_mean), '-b^')
        _ln3e = ax2.plot((stage,stage+1), (e_c0[e_m,e_j],e_c[e_m,e_j]), '-gv')
        #plot real value
        _ln1 = ax1.plot((stage,stage+1), (arr_prob[e_k,e_j]+0.002,arr_prob[e_k,e_j]+0.002), '-r')
        _ln2 = ax2.plot((stage,stage+1), (ul_mean, ul_mean), '-b')
        _ln3 = ax2.plot((stage,stage+1), (proc_mean[e_m,e_j],proc_mean[e_m,e_j]), '-g')
        #plot legend
        ax1.legend(_ln1e + _ln1 + _ln2e + _ln2 + _ln3e + _ln3,
                    ['Estimated Arrival Probability', 'Real Arrival Probability',
                    'Estimated Mean Uploading Time', 'Real Mean Uploading Time',
                    'Estimated Mean Processing Time', 'Real Mean Processing Time'
                    ], loc=0) #center right
        fig.tight_layout()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.3)
        #---------------------------------------------------------------------
        e_lambda0 = e_lambda; e_c0 = e_c; e_u0 = e_u
        stage += 1
        pass
    plt.ioff()
    plt.show()
    pass

def main_one_shot(args):
    np.random.seed( args.one_shot )
    record_folder = 'records-{prefix}/{postfix}-{tag}'.format(
                        prefix=RECORD_PREFIX, postfix=args.postfix, tag=args.one_shot)
    Path( record_folder ).mkdir(exist_ok=True, parents=True)
    #-----------------------------------------------------------
    stage = 0
    oldStat,   nowStat   = State(),          State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    SF_oldStat, SF_nowStat = State(), State()
    QA_oldStat, QA_nowStat = State(), State()
    RD_oldStat, RD_nowStat = State(), State()
    # default by reference
    TI_oldStat, TI_nowStat = oldStat, nowStat
    TI_Policy              = nowPolicy
    #-----------------------------------------------------------
    while stage < STAGE_ALT:
        # 1. one realization to next state
        val = None
        with Timer(output=True):
            arrivals = loadArrivalTrace(stage) #toss(arr_prob[k,j])
            br_delay = np.zeros((N_AP), dtype=np.int32)
            for k in range(N_AP):
                br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]
            #----------------------------------------------------------------
            systemStat     = (oldStat, nowStat, br_delay)
            oldPolicy      = nowPolicy
            if args.serial_flag:
                nowPolicy, val = serial_optimize(stage, systemStat, oldPolicy)
            else:
                nowPolicy, val = optimize(stage, systemStat, oldPolicy)
            #----------------------------------------------------------------
            oldStat,    nowStat    = nowStat,    NextState(arrivals, systemStat, oldPolicy, nowPolicy, None, None)
            systemStat             = (SF_oldStat, SF_nowStat, br_delay)
            SF_oldStat, SF_nowStat = SF_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, ASelfishPolicy, ASelfishPolicy)
            systemStat             = (QA_oldStat, QA_nowStat, br_delay)
            QA_oldStat, QA_nowStat = QA_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, AQueueAwarePolicy, AQueueAwarePolicy)
            systemStat             = (RD_oldStat, RD_nowStat, br_delay)
            RD_oldStat, RD_nowStat = RD_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, ARandomPolicy, ARandomPolicy)
            #----------------------------------------------------------------
            if stage < STAGE_EVAL:
                TI_Policy = nowPolicy
                TI_oldStat, TI_nowStat = oldStat, nowStat
            elif stage==STAGE_EVAL: #could integrated with 'stage<STAGE_EVAL' condition
                TI_Policy              = nowPolicy.copy()
                TI_oldStat, TI_nowStat = State().clone(oldStat), State().clone(nowStat)
            else:
                systemStat             = (TI_oldStat, TI_nowStat, br_delay)
                TI_oldStat, TI_nowStat = TI_nowStat, NextState(arrivals, systemStat, TI_Policy, TI_Policy, None, None)
            #----------------------------------------------------------------
            pass
        # 2. update the stage counter
        stage += 1
        # 3. record the stage (along this realization)
        stage_record_file = Path( record_folder, '%04d'%stage ).as_posix()
        with open(stage_record_file, 'wb') as fh:
            np.savez_compressed(fh, **{
                'MDP_value'   : val,
                'MDP_ap_stat' : nowStat.ap_stat,
                'MDP_es_stat' : nowStat.es_stat,
                'MDP_admissions': nowStat.admissions,
                'MDP_departures': nowStat.departures,
                #
                'Tight_ap_stat': TI_nowStat.ap_stat,
                'Tight_es_stat': TI_nowStat.es_stat,
                'Tight_admissions': TI_nowStat.admissions,
                'Tight_departures': TI_nowStat.departures,
                #
                'Selfish_ap_stat': SF_nowStat.ap_stat,
                'Selfish_es_stat': SF_nowStat.es_stat,
                'Selfish_admissions': SF_nowStat.admissions,
                'Selfish_departures': SF_nowStat.departures,
                #
                'QAware_ap_stat' : QA_nowStat.ap_stat,
                'QAware_es_stat' : QA_nowStat.es_stat,
                'QAware_admissions': QA_nowStat.admissions,
                'QAware_departures': QA_nowStat.departures,
                #
                'Random_ap_stat' : RD_nowStat.ap_stat,
                'Random_es_stat' : RD_nowStat.es_stat,
                'Random_admissions': RD_nowStat.admissions,
                'Random_departures': RD_nowStat.departures
            })
            pass
        print('one-shot-{}: stage {:04d}'.format(RECORD_PREFIX, stage))
        pass
    #-----------------------------------------------------------
    pass

def main_long_time(args):
    record_mark = 'records-{prefix}'.format(prefix=RECORD_PREFIX); print(record_mark)
    logger = getLogger(record_mark)
    record_folder = Path( record_mark, args.postfix )
    record_folder.mkdir(exist_ok=True, parents=True)
    if args.plot_flag:
        matplotlib.use("Qt5agg")
        plt.ion()
    
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
            #NOTE: load arrival trace from pre-defined trace folder (looped)
            arrivals = loadArrivalTrace(stage) #toss(arr_prob[k,j])
            # assert( np.any(arrivals==1) )

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
            nowStat        = NextState(arrivals, systemStat, oldPolicy, nowPolicy, None, None)
            #----------------------------------------------------------------
            systemStat             = (SF_oldStat, SF_nowStat, br_delay)
            SF_oldStat, SF_nowStat = SF_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, ASelfishPolicy, ASelfishPolicy)
            systemStat             = (QA_oldStat, QA_nowStat, br_delay)
            QA_oldStat, QA_nowStat = QA_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, AQueueAwarePolicy, AQueueAwarePolicy)
            systemStat             = (RD_oldStat, RD_nowStat, br_delay)
            RD_oldStat, RD_nowStat = RD_nowStat, NextState(arrivals, systemStat, NONE_POLICY, NONE_POLICY, ARandomPolicy, ARandomPolicy)
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

        stage_record = Path( record_folder, '%04d'%stage ).as_posix()
        np.savez(stage_record, **{
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
    summary_file = Path( record_folder, 'summary' ).as_posix()
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
        parser.add_argument('--one-shot', dest='one_shot', type=int, default=0,
            help='Run main_one_shot for fast averaging records.')
        parser.add_argument('--param-fit', dest='param_fit', action='store_true', default=False,
            help='Run main_param_fitting.')
        parser.add_argument('--serial-optimize', dest='serial_flag', action='store_true', default=False,
            help='Use serial optimization in MDP method.')
        parser.add_argument('--plot', dest='plot_flag', action='store_true', default=False,
            help='Plot figure (with Qt5) while running simulation.')
        parser.add_argument('--postfix', dest='postfix', type=str, default='test',
            help='specify postfix for record path/files.')
        parser.add_argument('--inject', dest='_', type=str, default='',
            help='always as last one for `params.py` usage.')
        args = parser.parse_args()

        if args.param_fit:
            main_param_fitting(args)
        elif args.one_shot!=0:
            main_one_shot(args)
        else:
            main_long_time(args)
    except Exception as e:
        raise e
    finally:
        pass

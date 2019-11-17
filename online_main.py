
import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib.pyplot as plt

def NextState(oldStat, nowStat, oldPolicy, nowPolicy, arrival_ap, br_delay):
    nextStat  = State().clone(nowStat)

    # update intermediate state with arrivals in each time slot 
    for n in range(N_SLT):
        TODO:
        pass

    return nextStat

def main():
    pathlib.Path('.logs').mkdir(exist_ok=True)
    pathlib.Path('figures').mkdir(exist_ok=True)

    stage = 0
    oldStat,   nowStat   = State(),          State()
    oldPolicy, nowPolicy = BaselinePolicy(), BaselinePolicy()
    
    while stage < STAGE:
        #toss job arrival for APs in each interval
        arrival_ap = np.zeros((N_SLT, N_AP, N_JOB), dtype=np.int32)
        for n in range(N_SLT):
            for j in range(N_JOB):
                for k in range(N_AP):
                    arrival_ap[n,k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

        # toss for broadcast delay for each AP
        br_delay = np.zeros((N_AP), dtype=np.int32)
        for k in range(N_AP):
            br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]

        # optimize and update the backup
        oldPolicy      = nowPolicy
        nowPolicy, val = optimize( oldStat, nowStat, oldPolicy, br_delay, stage )
        oldStat        = nowStat
        nowStat        = NextState(oldStat, nowStat, oldPolicy, nowPolicy, arrival_ap, br_delay)
        
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
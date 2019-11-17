
import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib.pyplot as plt

def NextState(stat, arrival_ap, oldPolicy, newPolicy):
    newStat  = State().clone(stat)

    # toss for broadcast delay for each AP
    br_delay = np.zeros((N_AP), dtype=np.int32)
    for k in range(N_AP):
        br_delay[k] = BR_RNG[ multoss(br_dist[k]) ]

    # update intermediate state with arrivals in each time slot 
    for n in range(N_SLT):
        TODO:
        pass

    return newStat

def main():
    pathlib.Path('.logs').mkdir(exist_ok=True)
    pathlib.Path('figures').mkdir(exist_ok=True)

    stage = 0
    stat  = State()
    oldPolicy, newPolicy = BaselinePolicy(), BaselinePolicy()
    
    while stage < STAGE:
        #toss job arrival for APs in each interval
        arrival_ap = np.zeros((N_SLT, N_AP, N_JOB), dtype=np.int32)
        for n in range(N_SLT):
            for j in range(N_JOB):
                for k in range(N_AP):
                    arrival_ap[n,k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

        oldPolicy      = newPolicy
        newPolicy, val = optimize(stat, oldPolicy)
        stat           = NextState(stat, arrival_ap, oldPolicy, newPolicy)
        
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
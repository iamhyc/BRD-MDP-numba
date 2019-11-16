
import pathlib
import numpy as np
from mdp import *
from params import*
from utility import *
import matplotlib.pyplot as plt

def NextState(stat, arrival_ap, oldPolicy, newPolicy):
    newStat = State().clone(stat)
    TODO:
    return newStat

def main():
    pathlib.Path('.logs').mkdir(exist_ok=True)
    pathlib.Path('figures').mkdir(exist_ok=True)

    stage = 0
    stat  = State()
    oldPolicy, newPolicy = BaselinePolicy(), BaselinePolicy()
    
    while stage < STAGE:
        arrival_ap = np.zeros((N_AP, N_JOB), dtype=np.int32)
        for j in range(N_JOB):
            for k in range(N_AP):
                arrival_ap[k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

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

import time
import pathlib
import numpy as np
from scipy.stats import norm
from numba import njit, prange

pathlib.Path('logs').mkdir(exist_ok=True)
pathlib.Path('figures').mkdir(exist_ok=True)

@njit
def multoss(p_vec):
    return (np.random.rand() > np.cumsum(p_vec)).argmin()

@njit
def toss(p):
    p_vec = np.array([1-p, p], dtype=np.float64)
    return multoss(p_vec)

@njit
def factorial(n):
    result = 1.0
    for i in range(1,n+1):
        result *= i
    return result

@njit
def binom(n, p, k):
    tmp  = factorial(n) / (factorial(k) * factorial(n-k))
    prob = tmp * np.power(p, k) * np.power(1-p, n-k)
    return prob

@njit
def FillMatDiagonal(mat, arr, offset=0):
    assert(mat.shape[0] == mat.shape[1]) #assert square matrix
    for i in range(len(arr)):
        if offset > 0:
            mat[i, i+offset] = arr[i]
        else:
            mat[i+offset, i] = arr[i]
    pass

@njit
def FillAColumn(mat, idx, arr, offset=0):
    for i in range(len(arr)):
        mat[idx, i+offset] = arr[i]
    pass

@njit
def FillARow(mat, idx, arr, offset=0):
    for i in range(len(arr)):
        mat[i+offset, idx] = arr[i]
    pass

@njit
def genFlatDist(size):          #e.g. [1, 1, ... 1, 1]
    arr = 0.1+0.1*np.random.rand(size).astype(np.float64)
    return (arr / np.sum(arr))

@njit
def genHeavyTailDist(size):     #e.g. [0, 0, ... 1, 1]
    mid_size = size - size//6
    arr_1 = 0.1*np.random.rand(mid_size).astype(np.float64)
    arr_2 = 0.6+0.1*np.random.rand(size-mid_size).astype(np.float64)
    arr = np.sort( np.concatenate((arr_1, arr_2)) )
    return (arr / np.sum(arr))

@njit
def genHeavyHeadDist(size):     #e.g. [1, 1, ... 0, 0]
    arr = genHeavyTailDist(size)
    return arr[::-1]

def genGaussianDist(size):      #e.g. [0, 0, ..1,1,1.., 0, 0]
    rv = norm(loc=size//2, scale=0.8)
    arr = rv.pdf(np.arange(size))
    arr = np.array(arr, dtype=np.float64)
    return (arr / np.sum(arr))

def genSplitDist(size):         #e.g. [1, 1, ..0,0,0.., 1, 1]
    arr = genGaussianDist(size)
    arr = 1 - arr
    return (arr / np.sum(arr))

def itest(param, test=True):
    if test:
        print(param)
        exit()
    pass

class Timer:
    def __init__(self, output=True):
        self.output = output
        self.delta = 0
        pass
    
    def __enter__(self):
        self._start = time.time()
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop = time.time()
        self.delta = self._stop - self._start
        if self.output:
            print('Time elapsed: %.3f'%self.delta)
        pass
    
    def start(self):
        self._start = time.time()

    def stop(self):
        self._stop = time.time()
        self.delta = self._stop - self._start
        return self.delta
    pass

import numpy as np
import pandas as pd
import numba
import scipy.stats

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_pairs(x, y):
    """Draw a pairs bootstrap sample."""
    inds = np.arange(len(x))
    bs_inds = draw_bs_sample(inds)

    return x[bs_inds], y[bs_inds]

def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def variance(data):
    n = len(data)
    mean = np.mean(data)
    total = 0
    for x in data:
        total += (x - mean)**2
    return total / (n * (n-1))

def ecdf(x, data):
    data_sorted = np.sort(data)
    index = 0
    ctr = 0
    ecdf_values = []
    for xx in np.sort(x):
        while index < len(data) and data_sorted[index] <= xx:
            ctr += 1
            index += 1
        ecdf_values.append(ctr/len(data))
    return ecdf_values
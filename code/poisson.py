import warnings
import numpy as np
import pandas as pd
import numpy.random
rg = numpy.random.default_rng()
import scipy.optimize
import scipy.stats as st
import tqdm
import bebi103
try:
    import multiprocess
except:
    import multiprocessing as multiprocess

def CDF_double_exp(beta_1, beta_2, t):
    frac = beta_1 * beta_2 / (beta_2 - beta_1)
    b1 = (1 - np.exp(-beta_1 * t)) / beta_1
    b2 = (1 - np.exp(-beta_2 * t)) / beta_2
    return frac * (b1 - b2)

def log_likelihood(n, beta_1, d_beta):
    like = []
    for t in n:
        p1 = np.log(beta_1 * (beta_1 + d_beta) / d_beta)
        p2 = -beta_1 * t
        p3 = np.log(1 - np.exp(-d_beta * t))
        like.append(p1 + p2 + p3)
    return like

def gen_microtubule(b1, db, size, rg):
    beta_1 = b1
    beta_2 = db + b1
    return rg.exponential(1/beta_1, size=size) + rg.exponential(1/beta_2, size=size)

def log_like_microtubule(params, n):
    """Log likelihood for the microtubule time to catastrophe measurements,
    parametrized by beta_1, d_beta."""
    beta_1, d_beta = params

    # limits:
    # beta_1 >= 0
    # d_beta >= 0
    if beta_1 < 0 or d_beta < 0:
        return -np.inf
    
    return np.sum(log_likelihood(n, beta_1, d_beta))


def mle_microtubule(n):
    """Perform maximum likelihood estimates for parameters for 
    the microtubule time to catastrophe measurements, parametrized by beta_1, d_beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_microtubule(params, n),
            x0=np.array([3, 3]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
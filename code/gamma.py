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

def log_like_gamma(params, n):
    """Log likelihood for gamma measurements, parametrized
    by alpha, beta."""
    alpha, beta = params

    # limits:
    # alpha > 0
    # beta > 0
    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, scale=1/beta))


def mle_gamma(n):
    """Perform maximum likelihood estimates for parameters for 
    gamma measurements, parametrized by alpha, beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_gamma(params, n),
            x0=np.array([3, 3]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))


def _draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False, rg=None):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if rg is None:
        rg = np.random.default_rng()
    
    if progress_bar:
#         tqdm.tqdm._instances.clear()
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, n_jobs=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    # Just call the original function if n_jobs is 1 (no parallelization)
    if n_jobs == 1:
        return _draw_bs_reps_mle(
            mle_fun, data, args=args, size=size, progress_bar=progress_bar)

    # Set up sizes of bootstrap replicates for each core, making sure we
    # get all of them, even if sizes % n_jobs != 0
    sizes = [size // n_jobs for _ in range(n_jobs)]
    sizes[-1] += size - sum(sizes)

    # Build arguments
    arg_iterable = [(mle_fun, data, args, s, progress_bar, None) for s in sizes]

    with multiprocess.Pool(n_jobs) as pool:
        result = pool.starmap(_draw_bs_reps_mle, arg_iterable)

    return np.concatenate(result)

def gen_gamma(alpha, beta, size, rg):
    return rg.gamma(alpha, scale=1/beta, size=size

def _draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False, rg=None,
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size, *args, rg)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.
    rg : numpy.random.Generator instance, default None
        RNG to be used in bootstrapping. If None, the default
        Numpy RNG is used with a fresh seed based on the clock.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if rg is None:
        rg = np.random.default_rng()

    params = mle_fun(data, *args)

    if progress_bar:
#         tqdm.tqdm._instances.clear()
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args, rg=rg)) for _ in iterator]
    )

def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, n_jobs=1, progress_bar=False
):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size, *args, rg)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    n_jobs : int, default 1
        Number of cores to use in drawing bootstrap replicates.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    # Just call the original function if n_jobs is 1 (no parallelization)
    if n_jobs == 1:
        return _draw_parametric_bs_reps_mle(
            mle_fun, gen_fun, data, args=args, size=size, progress_bar=progress_bar
        )

    # Set up sizes of bootstrap replicates for each core, making sure we
    # get all of them, even if sizes % n_jobs != 0
    sizes = [size // n_jobs for _ in range(n_jobs)]
    sizes[-1] += size - sum(sizes)

    # Build arguments
    arg_iterable = [(mle_fun, gen_fun, data, args, s, progress_bar, None) for s in sizes]

    with multiprocess.Pool(n_jobs) as pool:
        result = pool.starmap(_draw_parametric_bs_reps_mle, arg_iterable)

    return np.concatenate(result)
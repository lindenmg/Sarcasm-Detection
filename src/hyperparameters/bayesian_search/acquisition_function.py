"""
This file contains the acquisition functions,
that may be used for bayesian optimisation
"""
import numpy as np
from scipy.stats import norm


def expected_improvement(x, gaussian_process, evaluated_performance, greater_is_better=False):
    """ expected_improvement

    Expected improvement acquisition function.
    Original implementation: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_performance: Numpy array.
            Numpy array that contains the values of the performance function (e.g. loss) for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the performance function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    mu, sigma = gaussian_process.predict(x, return_std=True)

    if greater_is_better:
        performance_optimum = np.max(evaluated_performance)
    else:
        performance_optimum = np.min(evaluated_performance)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        z = scaling_factor * (mu - performance_optimum) / sigma
        ei = scaling_factor * (mu - performance_optimum) * norm.cdf(z) + sigma * norm.pdf(z)

    return ei

import numpy as np


def GBM_sample_paths(S0, r, sigma, T, num_sim, num_time_steps):

    """
    Simulate Geometric Brownian Motion sample paths.
    :param S0: Initial stock price.
    :param r: Risk-free interest rate.
    :param sigma: Volatility.
    :param T: Maturity.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps in the discretisation of T.
    :return: Geometric Brownian Motion sample paths with time index.
    """

    h = np.divide(T, num_time_steps)
    normal_rvs = np.multiply(np.sqrt(h), np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim)))
    S = np.ones((num_time_steps+1, num_sim))
    S[0, :] = np.ones(num_sim) * S0
    time_steps = [0]
    for i in range(1, num_time_steps+1):
        S[i, :] = S[i-1, :] + np.multiply(S[i-1, :], r * h + np.multiply(sigma, normal_rvs[i-1]))
        time_steps.append(i * h)
    return np.concatenate((S[:, :, None],
                                   np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                             axis=1)),
                          axis=2)
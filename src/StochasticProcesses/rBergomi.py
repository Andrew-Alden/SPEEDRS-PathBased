import numpy as np

# Code to simulate the fractional Brownian Motion (i.e. method fBm_path_rDonsker) was adapted from Github repo: https://github.com/amuguruza/RoughFCLT.
def fBm_path_rDonsker(Z1, H, T, dt, num_sim, num_time_steps, kernel="optimal"):

    """
    Simulate the Volterra process.
    :param Z1: Normal distributed random variables. Array of shape [Num Sim, Num Time Steps + 1]
    :param H: Hurst parameter.
    :param T: Maturity.
    :param dt: Length of time steps.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps.
    :param kernel: String describing kernel operator.
    :return: Volterra paths. Array of shape [Num Time Steps + 1, Num Sim].
    """

    assert 0 < H < 1.0

    dW = np.power(dt, H) * Z1

    i = np.arange(num_time_steps) + 1
    if kernel == "optimal":
        opt_k = np.power((np.power(i, 2 * H) - np.power(i - 1., 2 * H)) / 2.0 / H, 0.5)
    elif kernel == "naive":
        opt_k = np.power(i, H - 0.5)
    else:
        raise NameError("That was not a valid kernel")

    Y = np.zeros([num_sim, num_time_steps + 1])
    for i in range(int(num_sim)):
        Y[i, 1:num_time_steps + 1] = np.convolve(opt_k, dW[i, :])[0:num_time_steps]

    return np.transpose(Y * np.power(T, H))


def r_bergomi_Phi(V0, xi_0, GFO, C, nu, H, dt, num_time_steps, num_sim):

    """
    Simulate the volatility process.
    :param V0: Initial volatility.
    :param xi_0: Forward variance curve. Array of shape [Number of Time Steps + 1].
    :param GFO: Volterra paths. Array of shape [Num Time Steps + 1, Num Sim].
    :param C: Parameter C.
    :param nu: Parameter nu.
    :param H: Hurst parameter.
    :param dt: Length of time steps.
    :param num_time_steps: Number of time steps.
    :param num_sim: Number of simulation.
    :return: Volatility process. Array of shape [Num Time Steps + 1, Num Sim].
    """

    V = np.ones((num_time_steps + 1, num_sim))
    V[0, :] = np.ones(num_sim) * V0
    variance = lambda t: np.power(t, 2 * H) / (2 * H)

    for i in range(1, num_time_steps + 1):
        V[i, :] = np.multiply(xi_0[i] * np.exp(-2 * (nu ** 2) * (C ** 2) * variance(dt * i)),
                              np.exp(np.multiply(2 * nu * C, GFO[i, :])))

    return V

def r_bergomi_sample_paths_functional_central_limit(S0, V0, H, xi_0, C, nu, rho, T, num_time_steps, num_sim):

    """
    Generate rBergomi sample paths.
    :param S0: Spot value.
    :param V0: Initial volatility.
    :param H: Hurst parameter.
    :param xi_0: Forward variance curve. Array of shape [Num Time Steps + 1].
    :param C: Parameter C.
    :param nu: Parameter nu.
    :param rho: Correlation parameter.
    :param T: Maturity.
    :param num_time_steps: Number of discretisation time steps.
    :param num_sim: Number of simulations.
    :return: 1) Log-sample paths. Array of shape [Num Time Steps + 1, Num Sim].
             2) Time steps. Array of shape [Num Time Steps + 1].
             3) Sample paths and time steps. Array of shape [Num Time Steps + 1, Num Sim, 2].
    """

    assert num_sim % 2 == 0

    dt = T / (num_time_steps)
    dt_2 = 1 / num_time_steps

    zeta = np.random.normal(loc=0, scale=1, size=(num_time_steps + 1, int(num_sim / 2)))
    neg_zeta = -1 * zeta
    Z1 = np.concatenate((zeta, neg_zeta), axis=1)

    xi = np.random.normal(loc=0, scale=1, size=(num_time_steps + 1, int(num_sim / 2)))

    rho_bar = np.sqrt(1 - rho ** 2)
    antithetic_variate_1 = rho * xi + rho_bar * zeta
    antithetic_variate_2 = rho * xi + rho_bar * neg_zeta

    Z2 = np.concatenate((antithetic_variate_1, antithetic_variate_2), axis=1)

    GFO = fBm_path_rDonsker(np.transpose(Z1), H, T, dt_2, num_sim, num_time_steps, kernel='optimal')

    X = np.zeros((num_time_steps + 1, num_sim))
    X[0, :] = np.ones(num_sim) * np.log(S0)

    time_steps = [0]

    V = r_bergomi_Phi(V0, xi_0, GFO, C, nu, H, dt, num_time_steps, num_sim)

    for i in range(1, num_time_steps + 1):

        X[i, :] = X[i-1, :] - 0.5*dt*V[i-1, :] + np.sqrt(dt)*np.multiply(np.sqrt(V[i-1, :]), Z2[i-1])

        time_steps.append(i * dt)

    return X, time_steps, np.concatenate((np.exp(X)[:, :, None],
                                                 np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                                           axis=1)),
                                         axis=2)


def conditional_r_bergomi_sample_paths_functional_central_limit(S0, V0, H, xi_0, C, nu, rho, T, num_time_steps, num_sim):

    """
    Generate conditional rBergomi sample paths.
    :param S0: Spot value.
    :param V0: Initial volatility.
    :param H: Hurst parameter.
    :param xi_0: Forward variance curve. Array of shape [Num Time Steps + 1].
    :param C: Parameter C.
    :param nu: Parameter nu.
    :param rho: Correlation parameter.
    :param T: Maturity.
    :param num_time_steps: Number of discretisation time steps.
    :param num_sim: Number of simulations.
    :return: 1) Log-sample paths. Array of shape [Num Time Steps + 1, Num Sim].
             2) Time steps. Array of shape [Num Time Steps + 1].
             3) Sample paths and time steps. Array of shape [Num Time Steps + 1, Num Sim, 2].
    """

    assert num_sim % 2 == 0

    dt = T / (num_time_steps)
    dt_2 = 1 / num_time_steps

    zeta = np.random.normal(loc=0, scale=1, size=(num_time_steps + 1, int(num_sim / 2)))
    neg_zeta = -1 * zeta
    Z1 = np.concatenate((zeta, neg_zeta), axis=1)

    xi = np.random.normal(loc=0, scale=1, size=(num_time_steps + 1, int(num_sim / 2)))

    rho_bar = np.sqrt(1 - rho ** 2)
    antithetic_variate_1 = rho * xi + rho_bar * zeta
    antithetic_variate_2 = rho * xi + rho_bar * neg_zeta

    Z2 = np.concatenate((antithetic_variate_1, antithetic_variate_2), axis=1)

    GFO = fBm_path_rDonsker(np.transpose(Z1), H, T, dt_2, num_sim, num_time_steps, kernel='optimal')

    X = np.zeros((num_time_steps + 1, num_sim))
    X[0, :] = np.ones(num_sim) * np.log(S0)

    time_steps = [0]

    V = r_bergomi_Phi(V0, xi_0, GFO, C, nu, H, dt, num_time_steps, num_sim)

    Sigma = (1 - rho ** 2) * dt * np.sum(V, axis=0)
    for i in range(1, num_time_steps + 1):

        X[i, :] = X[0, :] - (rho ** 2 / 2) * dt * np.sum(V[:i, :], axis=0) \
                  + rho * np.sqrt(dt) * (np.sum(np.multiply(Z2[:i], np.sqrt(V[:i, :])), axis=0))

        time_steps.append(i * dt)

    return X, time_steps, Sigma, np.concatenate((np.exp(X)[:, :, None],
                                                 np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                                           axis=1)),
                                                axis=2)






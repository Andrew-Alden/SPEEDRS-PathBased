# Adapted from https://github.com/maudl3116/Distribution_Regression_Streams

import numpy as np
from src.StochasticProcesses.rBergomi import r_bergomi_sample_paths_functional_central_limit, conditional_r_bergomi_sample_paths_functional_central_limit
from src.StochasticProcesses.Heston import heston_sample_paths_inv
from tqdm import tqdm


def bags_to_2D(input_):
    '''
    This function transforms input data in the form of bags of items, where each item is a D-dimensional time series
    (represented as a list of list of (T,D) matrices, where T is the length of the time series) into a 2D array to be
    compatible with what sklearn pipeline takes in input, i.e. a 2D array (n_samples, n_features).

    (1) Pad each time series with its last value such that all time series have the same length.
    -> This yields lists of lists of 2D arrays (max_length,dim)
    (2) Stack the dimensions of the time series for each item
    -> This yields lists of lists of 1D arrays (max_length*dim)
    (3) Stack the items in a bag
    -> This yields lists of 1D arrays (n_item*max_length*dim)
    (4) Create "dummy items" which are time series of NaNs, such that they can be retrieved and removed at inference time
    -> This yields a 2D array (n_bags,n_max_items*max_length*dim)

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)

       Output: a 2D array of shape (n_bags,n_max_items x max_length x dim)

    '''

    # dimension of the state space of the time series (D)
    dim_path = input_[0][0].shape[1]

    # Find the maximum length to be able to pad the smaller time-series
    T = [e[0].shape[0] for e in input_]
    common_T = max(T)

    new_list = []
    for bag in input_:
        new_bag = []
        for item in bag:
            # (1) if the time series is smaller than the longest one, pad it with its last value
            if item.shape[0] < common_T:
                new_item = np.concatenate([item, np.repeat(item[-1, :][None, :], common_T - item.shape[0], axis=0)])
            else:
                new_item = item
            new_bag.append(new_item)
        new_bag = np.array(new_bag)
        # (2) stack the dimensions for all time series in a bag
        new_bag = np.concatenate([new_bag[:, :, k] for k in range(dim_path)], axis=1)
        new_list.append(new_bag)

    # (3) stack the items in each bag
    items_stacked = [bag.flatten() for bag in new_list]

    # retrieve the maximum number of items
    max_ = [bag.shape for bag in items_stacked]
    max_ = np.max(max_)
    max_items = int(max_ / (dim_path * common_T))

    # (4) pad the vectors with nan items such that we can obtain a 2d array.
    items_naned = [np.append(bag, (max_ - len(bag)) * [np.nan]) for bag in items_stacked]  # np.nan

    X = np.array(items_naned)

    return X, max_items, common_T, dim_path


def rbf_mmd_mat(X, Y, gamma=1.0, max_items=None, sym=False):
    M = max_items

    alpha = 1. / (2 * gamma ** 2)

    if sym:

        XX = np.dot(X, X.T)
        X_sqnorms = np.diagonal(XX)
        K_XX = np.exp(-alpha * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))

        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]

        K_XY_means = np.nanmean(K_XX.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))
        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means

    else:

        XX = np.dot(X, X.T)
        XY = np.dot(X, Y.T)
        YY = np.dot(Y, Y.T)

        X_sqnorms = np.diagonal(XX)
        Y_sqnorms = np.diagonal(YY)

        K_XY = np.exp(-alpha * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = np.exp(-alpha * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = np.exp(-alpha * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        # blocks of bags
        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_YY_blocks = [K_YY[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_YY.shape[0] // M)]

        # nanmeans
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]  # n_bags_test
        K_YY_means = [np.nanmean(bag) for bag in K_YY_blocks]  # n_bags_train

        K_XY_means = np.nanmean(K_XY.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))

        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_YY_means)[np.newaxis, :] - 2 * K_XY_means

    return mmd


def rbf_matern_mmd_mat(X, Y, gamma=1.0, max_items=None, sym=False):
    M = max_items

    rho = 1. / gamma
    sqrt3 = np.sqrt(3.0)

    if sym:

        # Matern 3/2
        XX = np.dot(X, X.T)
        X_sqnorms = np.diagonal(XX)
        r2 = -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]
        r = np.sqrt(r2)
        K_XX = (1.0 + sqrt3 * rho * r) * np.exp(-sqrt3 * rho * r)

        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]

        K_XY_means = np.nanmean(K_XX.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))
        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means

    else:

        # Matern 3/2
        XX = np.dot(X, X.T)
        XY = np.dot(X, Y.T)
        YY = np.dot(Y, Y.T)

        X_sqnorms = np.diagonal(XX)
        Y_sqnorms = np.diagonal(YY)

        r2_XX = -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]
        r_XX = np.sqrt(r2_XX)

        r2_YY = -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
        r_YY = np.sqrt(r2_YY)

        r2_XY = -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
        r_XY = np.sqrt(r2_XY)

        K_XY = (1.0 + sqrt3 * rho * r_XY) * np.exp(-sqrt3 * rho * r_XY)
        K_XX = (1.0 + sqrt3 * rho * r_XX) * np.exp(-sqrt3 * rho * r_XX)
        K_YY = (1.0 + sqrt3 * rho * r_YY) * np.exp(-sqrt3 * rho * r_YY)


        # blocks of bags
        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_YY_blocks = [K_YY[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_YY.shape[0] // M)]

        # nanmeans
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]  # n_bags_test
        K_YY_means = [np.nanmean(bag) for bag in K_YY_blocks]  # n_bags_train

        K_XY_means = np.nanmean(K_XY.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))

        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_YY_means)[np.newaxis, :] - 2 * K_XY_means

    return mmd


def construct_pointwise_kernel_features(base_models, rbergomi_heston_dict, kernel_fn, num_sim=400,
                                        num_time_steps=14, T=1.0, sigma=1, num_samples=2000, inc=1.0, alpha_list=[-1]):

    """
    :param base_models: Dictionary of base process paths.
    :param rbergomi_heston_dict: Dataset dictionary.
    :param kernel_fn: Pointwise kernel.
    :param num_sim: Number of simulations. Default is 400.
    :param num_time_steps: Number of time steps. Default is 14.
    :param T: Terminal time. Default is 1.0.
    :param sigma: Kernel parameter. Default is 1.0.
    :param num_samples: Number of samples in dataset. Default is 2000.
    :param inc: Increment for dictionary counter.
    :param alpha_list: List of alpha values if alpha is contained within the dataset dictionary. If the list is [-1],
                       then look for alpha value in data dictionary. Default is [-1].
    :return: Kernel-based baseline features.
    """
    
    rbergomi_heston_rbf_kernel_features = []

    base_processes_list = base_models['base_rbergomi_paths'] + base_models['base_heston_paths'] + base_models[
        'base_gbm_paths']

    for i in range(len(base_processes_list)):
        base_processes_list[i] = base_processes_list[i].transpose((1, 0, 2))[:num_sim, ...]

    for i in tqdm(range(num_samples)):

        p1 = r_bergomi_sample_paths_functional_central_limit(rbergomi_heston_dict['S0_rbergomi'][i * inc],
                                                             rbergomi_heston_dict['v0_rbergomi_list'][i * inc],
                                                             rbergomi_heston_dict['H_list'][i * inc],
                                                             [rbergomi_heston_dict['xi_0_list'][i * inc] for _ in
                                                              range(num_time_steps + 1)],
                                                             np.sqrt(2 * rbergomi_heston_dict['H_list'][i * inc]),
                                                             rbergomi_heston_dict['nu_list'][i * inc],
                                                             rbergomi_heston_dict['rho_rbergomi_list'][i * inc],
                                                             T,
                                                             num_time_steps,
                                                             num_sim)[-1]

        p2 = heston_sample_paths_inv(rbergomi_heston_dict['S0_rbergomi'][i * inc],
                                     rbergomi_heston_dict['v0_heston_list'][i * inc],
                                     rbergomi_heston_dict['r_rbergomi'][i * inc],
                                     rbergomi_heston_dict['rho_heston_list'][i * inc],
                                     rbergomi_heston_dict['mean_vol_list'][i * inc],
                                     rbergomi_heston_dict['speed_list'][i * inc],
                                     rbergomi_heston_dict['vol_of_vol_list'][i * inc],
                                     T, num_sim, num_time_steps)[0]

        for alpha in alpha_list:

            if alpha == -1:
                alpha = rbergomi_heston_dict['alpha_list'][i]

            p = np.add(alpha * p1, (1 - alpha) * p2)

            p[:, :, 0] /= rbergomi_heston_dict['S0_rbergomi'][i * inc]

            p = p.transpose((1, 0, 2))

            pp, max_items, common_T, dim_path = bags_to_2D([p] + base_processes_list)

            size_item = dim_path * common_T

            f = []

            for i in range(1, 21):
                f.append(kernel_fn(pp[0].reshape(-1, size_item), pp[i].reshape(-1, size_item), max_items=max_items,
                                   gamma=sigma)[0][0])

        rbergomi_heston_rbf_kernel_features.append(f)

    return rbergomi_heston_rbf_kernel_features
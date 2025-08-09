import numpy as np
import torch.cuda
from tqdm.auto import tqdm
import torch
import iisignature

from src.StochasticProcesses.rBergomi import r_bergomi_sample_paths_functional_central_limit
from src.StochasticProcesses.Heston import heston_sample_paths_inv
from src.StochasticProcesses.BlackScholes import GBM_sample_paths

from src.utils import save_path_params


def save_checkpoint(xi_0_list, nu_list, rho_rbergomi_list, H_list, v0_rbergomi_list, vol_of_vol_list,
                    speed_list, mean_volatility_list, v0_heston_list, rho_heston_list, sigma_list,
                    r_list, rbergomi_rbergomi_sample_paths, rbergomi_rbergomi_distances, rbergomi_heston_sample_paths,
                    rbergomi_heston_distances, rbergomi_gbm_sample_paths, rbergomi_gbm_distances,
                    heston_heston_sample_paths, heston_heston_distances, heston_gbm_sample_paths,
                    heston_gbm_distances, gbm_gbm_sample_paths, gbm_gbm_distances,
                    filename='MMDApproxData/Datasets/mmd_data_dict'):
    """
    Save checkpoint.
    :param xi_0_list: List of forward variance curve parameter values.
    :param nu_list: List of nu parameter values.
    :param rho_rbergomi_list: List of rBergomi correlation parameter values.
    :param H_list: List of Hurst parameter values.
    :param v0_rbergomi_list: List of rBergomi initial volatility parameter values.
    :param vol_of_vol_list: List of vol of vol parameter values.
    :param speed_list: List of speed parameter values.
    :param mean_volatility_list: List of mean vol parameter values.
    :param v0_heston_list: List of Heston initial volatility parameter values.
    :param rho_heston_list: List of Heston correlation parameter values.
    :param sigma_list: List of Black-Scholes volatility parameter values.
    :param r_list: List of drift parameter values.
    :param rbergomi_rbergomi_sample_paths: List of (rBergomi, rBergomi) sample paths.
    :param rbergomi_rbergomi_distances: List of (rBergomi, rBergomi) distances.
    :param rbergomi_heston_sample_paths: List of (rBergomi, Heston) sample paths.
    :param rbergomi_heston_distances: List of (rBergomi, Heston) distances.
    :param rbergomi_gbm_sample_paths: List of (rBergomi, GBM) sample paths.
    :param rbergomi_gbm_distances: List of (rBergomi, GBM) distances.
    :param heston_heston_sample_paths: List of (Heston, Heston) sample paths.
    :param heston_heston_distances: List of (Heston, Heston) distances.
    :param heston_gbm_sample_paths: List of (Heston, GBM) sample paths.
    :param heston_gbm_distances: List of (Heston, GBM) distances.
    :param gbm_gbm_sample_paths: List of (GBM, GBM) sample paths.
    :param gbm_gbm_distances: List of (GBM, GBM) distances.
    :param filename: File name. Default is 'MMDApproxData/Datasets/mmd_data_dict'.
    """

    mmd_data_dict = {
        'xi_0_list': xi_0_list,
        'nu_list': nu_list,
        'rho_rbergomi_list': rho_rbergomi_list,
        'H_list': H_list,
        'v0_rbergomi_list': v0_rbergomi_list,
        'vol_of_vol_list': vol_of_vol_list,
        'speed_list': speed_list,
        'mean_volatility_list': mean_volatility_list,
        'v0_heston_list': v0_heston_list,
        'rho_heston_list': rho_heston_list,
        'sigma_list': sigma_list,
        'r_list': r_list,
        'rbergomi_rbergomi_sample_paths': rbergomi_rbergomi_sample_paths,
        'rbergomi_rbergomi_distances': rbergomi_rbergomi_distances,
        'rbergomi_heston_sample_paths': rbergomi_heston_sample_paths,
        'rbergomi_heston_distances': rbergomi_heston_distances,
        'rbergomi_gbm_sample_paths': rbergomi_gbm_sample_paths,
        'rbergomi_gbm_distances': rbergomi_gbm_distances,
        'heston_heston_sample_paths': heston_heston_sample_paths,
        'heston_heston_distances': heston_heston_distances,
        'heston_gbm_sample_paths': heston_gbm_sample_paths,
        'heston_gbm_distances': heston_gbm_distances,
        'gbm_gbm_sample_paths': gbm_gbm_sample_paths,
        'gbm_gbm_distances': gbm_gbm_distances,
        'T': 1.0
    }

    save_path_params(mmd_data_dict, filename)


class MMDDataset:

    """
    Class handling the generation of the rank-2 sig-MMD approximator dataset.
    """

    def __init__(self, signature_kernel, num_r_bergomi_pairs, num_r_bergomi_heston_pairs,
                 num_r_bergomi_gbm_pairs, num_heston_pairs, num_heston_gbm_pairs, num_gbm_pairs, device, S0=1.0,
                 T=1.0, num_time_steps=14, num_sim=400, xi_0_range=(0.01, 0.2), nu_range=(0.5, 4.0),
                 rho_range=(-1, 1), H_range=(0.025, 0.5), vol_of_vol_range=(0.2, 0.8),
                 speed_range=(0.2, 0.8), mean_volatility_range=(0.2, 0.8), v0_range=(0.2, 0.8),
                 sigma_range=(0.2, 0.8), r_range=(0.01, 0.2)):

        """
        Contructor.
        :param signature_kernel: Signature kernel object used to compute the sig-MMD.
        :param num_r_bergomi_pairs: Number of MMD values between process pairs (rBergomi, rBergomi).
        :param num_r_bergomi_heston_pairs: Number of MMD values between process pairs (rBergomi, Heston).
        :param num_r_bergomi_gbm_pairs: Number of MMD values between process pairs (rBergomi, GBM).
        :param num_heston_pairs: Number of MMD values between process pairs (Heston, Heston).
        :param num_heston_gbm_pairs: Number of MMD values between process pairs (Heston, GBM).
        :param num_gbm_pairs: Number of MMD values between process pairs (GBM, GBM).
        :param device: Device on which to compute sig-MMD.
        :param S0. Initial path value. Default is 1.0.
        :param T. Terminal value. Default is 1.0.
        :param num_time_steps: Number of time steps (excluding initial value (S0)). Default is 14.
        :param num_sim: Number of simulations (batch size). Default is 400.
        :param xi_0_range: Range of possible values for parameter xi_0. Default is (0.01, 0.2).
        :param nu_range: Range of possible values for parameter nu. Default is (0.5, 4.0).
        :param rho_range: Range of possible values for parameter rho. Default is (-1.0, 1.0).
        :param H_range: Range of possible values for parameter H. Default is (0.025, 0.5).
        :param vol_of_vol_range: Range of possible values for parameter vol_of_vol. Default is (0.2, 0.8).
        :param speed_range: Range of possible values for parameter speed. Default is (0.2, 0.8).
        :param mean_volatility_range: Range of possible values for parameter mean_volatility. Default is (0.2, 0.8).
        :param v0_range: Range of possible values for parameter v0. Default is (0.2, 0.8).
        :param sigma_range: Range of possible values for parameter sigma. Default is (0.2, 0.8).
        :param r_range: Range of possible values for parameter r. Default is (0.01, 0.2).
        """

        self.signature_kernel = signature_kernel

        self.num_r_bergomi_pairs = num_r_bergomi_pairs
        self.num_r_bergomi_heston_pairs = num_r_bergomi_heston_pairs
        self.num_r_bergomi_gbm_pairs = num_r_bergomi_gbm_pairs
        self.num_heston_pairs = num_heston_pairs
        self.num_heston_gbm_pairs = num_heston_gbm_pairs
        self.num_gbm_pairs = num_gbm_pairs

        self.device = device

        self.S0 = S0
        self.T = T
        self.num_time_steps = num_time_steps
        self.num_sim = num_sim

        self.xi_0_range = xi_0_range
        self.nu_range = nu_range
        self.rho_range = rho_range
        self.H_range = H_range
        self.vol_of_vol_range = vol_of_vol_range
        self.speed_range = speed_range
        self.mean_volatility_range = mean_volatility_range
        self.v0_range = v0_range
        self.sigma_range = sigma_range
        self.r_range = r_range

        self.rbergomi_rbergomi_sample_paths = []
        self.rbergomi_rbergomi_distances = []

        self.rbergomi_heston_sample_paths = []
        self.rbergomi_heston_distances = []

        self.rbergomi_gbm_sample_paths = []
        self.rbergomi_gbm_distances = []

        self.heston_heston_sample_paths = []
        self.heston_heston_distances = []

        self.heston_gbm_sample_paths = []
        self.heston_gbm_distances = []

        self.gbm_gbm_sample_paths = []
        self.gbm_gbm_distances = []

        self.xi_0_list = []
        self.nu_list = []
        self.rho_rbergomi_list = []
        self.H_list = []
        self.v0_rbergomi_list = []

        self.vol_of_vol_list = []
        self.speed_list = []
        self.mean_volatility_list = []
        self.v0_heston_list = []
        self.rho_heston_list = []

        self.r_list = []

        self.sigma_list = []

    def generate_rand_rbergomi(self):

        """
        Generate a random rBergomi path.
        """

        p_xi_0 = np.random.uniform(self.xi_0_range[0], self.xi_0_range[1])
        p_nu = np.random.uniform(self.nu_range[0], self.nu_range[1])
        p_rho = np.random.uniform(self.rho_range[0], self.rho_range[1])
        p_H = np.random.uniform(self.H_range[0], self.H_range[1])
        p_v0 = np.random.uniform(self.v0_range[0], self.v0_range[1])

        path = r_bergomi_sample_paths_functional_central_limit(self.S0,
                                                               p_v0,
                                                               p_H,
                                                               [p_xi_0 for _ in range(self.num_time_steps + 1)],
                                                               np.sqrt(2 * p_H),
                                                               p_nu,
                                                               p_rho,
                                                               self.T,
                                                               self.num_time_steps,
                                                               self.num_sim)[-1]

        self.xi_0_list.append(p_xi_0)
        self.nu_list.append(p_nu)
        self.rho_rbergomi_list.append(p_rho)
        self.H_list.append(p_H)
        self.v0_rbergomi_list.append(p_v0)

        return path

    def generate_rand_heston(self):

        """
        Generate a random Heston path.
        """

        p_v0 = np.random.uniform(self.v0_range[0], self.v0_range[1])
        p_r = np.random.uniform(self.r_range[0], self.r_range[1])
        p_rho = np.random.uniform(self.rho_range[0], self.rho_range[1])
        p_mean_vol = np.random.uniform(self.mean_volatility_range[0], self.mean_volatility_range[1])
        p_speed = np.random.uniform(self.speed_range[0], self.speed_range[1])
        p_vol_of_vol = np.random.uniform(self.vol_of_vol_range[0], self.vol_of_vol_range[1])

        path = heston_sample_paths_inv(self.S0,
                                       p_v0,
                                       p_r,
                                       p_rho,
                                       p_mean_vol,
                                       p_speed,
                                       p_vol_of_vol,
                                       self.T,
                                       self.num_sim,
                                       self.num_time_steps)[0]

        self.vol_of_vol_list.append(p_vol_of_vol)
        self.speed_list.append(p_speed)
        self.mean_volatility_list.append(p_mean_vol)
        self.v0_heston_list.append(p_v0)
        self.rho_heston_list.append(p_rho)
        self.r_list.append(p_r)

        return path

    def generate_rand_gbm(self):

        """
        Generate a random GBM path.
        """

        p_sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        p_r = np.random.uniform(self.r_range[0], self.r_range[1])

        path = GBM_sample_paths(self.S0, p_r, p_sigma, self.T, self.num_sim, self.num_time_steps)

        self.sigma_list.append(p_sigma)
        self.r_list.append(p_r)

        return path

    def compute_distance_path_pairs(self):

        """
        Generate the paths and rank-2 sig-MMD distances.
        """

        for _ in tqdm(range(self.num_r_bergomi_pairs)):
            p1 = self.generate_rand_rbergomi()
            p2 = self.generate_rand_rbergomi()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.rbergomi_rbergomi_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.rbergomi_rbergomi_distances.append(d)

        for _ in tqdm(range(self.num_r_bergomi_heston_pairs)):
            p1 = self.generate_rand_rbergomi()
            p2 = self.generate_rand_heston()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.rbergomi_heston_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.rbergomi_heston_distances.append(d)

        for _ in tqdm(range(self.num_r_bergomi_gbm_pairs)):
            p1 = self.generate_rand_rbergomi()
            p2 = self.generate_rand_gbm()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.rbergomi_gbm_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.rbergomi_gbm_distances.append(d)

        for _ in tqdm(range(self.num_heston_pairs)):
            p1 = self.generate_rand_heston()
            p2 = self.generate_rand_heston()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.heston_heston_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.heston_heston_distances.append(d)

        for _ in tqdm(range(self.num_heston_gbm_pairs)):
            p1 = self.generate_rand_heston()
            p2 = self.generate_rand_gbm()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.heston_gbm_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.heston_gbm_distances.append(d)

        for _ in tqdm(range(self.num_gbm_pairs)):
            p1 = self.generate_rand_gbm()
            p2 = self.generate_rand_gbm()

            path1 = torch.transpose(torch.from_numpy(p1), 0, 1).to(device=self.device)
            path2 = torch.transpose(torch.from_numpy(p2), 0, 1).to(device=self.device)

            d = self.signature_kernel.compute_mmd(path1, path2, lambda_=1e-5, estimator='ub', order=2).cpu().item()

            self.gbm_gbm_sample_paths.append([np.asarray(p1), np.asarray(p2)])
            self.gbm_gbm_distances.append(d)

    def compute_expected_sigs(self, sigma=1, sig_level=4):

        """
        Compute the expected signatures.
        :param sigma: Sigma hyperparameter. Default is 1.
        :param sig_level: Signature truncation level. Default is 4.
        :return: List of expected signatures.
        """

        sample_paths = np.concatenate((np.asarray(self.rbergomi_rbergomi_sample_paths),
                                       np.asarray(self.rbergomi_heston_sample_paths),
                                       np.asarray(self.rbergomi_gbm_sample_paths),
                                       np.asarray(self.heston_heston_sample_paths),
                                       np.asarray(self.heston_gbm_sample_paths),
                                       np.asarray(self.gbm_gbm_sample_paths)), axis=0)

        expceted_sigs = np.mean(iisignature.sig(
            np.exp(-np.divide(np.power(np.transpose(np.asarray(sample_paths), (0, 1, 3, 2, 4)), 2), sigma)), sig_level),
                                axis=2)
        self.expceted_sigs = expceted_sigs.reshape(*expceted_sigs.shape[:-2], -1)
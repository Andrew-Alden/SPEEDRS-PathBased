from lightgbm import LGBMRegressor
import lightgbm as lgb
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
from torch import nn
import torch.cuda

from src.RegressionModel.train_utils import splice_features
from src.RegressionModel.regression_scaler_model import RegressionModelWithScaler
from src.utils import load_path_params

def load_lgbm_model(model_file, N):

    """
    Load the best LightGBM model.
    :param model_file: String containing the file name.
    :param N: Number of base processes.
    :return: The LightGBM model.
    """

    booster = lgb.Booster(model_file=model_file)
    model = LGBMRegressor()
    model._Booster = booster
    model.fitted_ = True
    model._n_features = N
    model._n_features_in_ = N
    return model


def test_neural_net(architecture, best_model_filename, scaler_filename, features, N, device, suffix='', num_runs=5,
                    splicing_dict={'5': [(0, 2), (8, 10), (16, 17)], '10': [(0, 4), (8, 12), (16, 18)],
                                   '20': [(0, 20)]}):

    """
    Test neural network model.
    :param architecture: Architecture of the model.
    :param best_model_filename: Filename corresponding to the best model.
    :param scaler_filename: Filename corresponding to the fitter scaler.
    :param features: List of features to be used as input to the neural network.
    :param N: Number of base processes.
    :param suffix: Suffix string which describes the model. Used to load the best model. Default is ''.
    :param num_runs: Number of independent models over which to perform the test. Default is 5.
    :param splicing_dict: Dictionary describing splicing procedure. Each key contains a list of tuples
                          with each tuple containing the start index and end index used to splice the array.
    :return: Dictionary of predictions. Keys correspond to model number and values are the model predictions.
    """

    predictions_dict = {}

    model_inputs = splice_features(features, N, splicing_dict)

    for i in range(num_runs):
        pricing_model = RegressionModelWithScaler(architecture, None, None, None, device,
                                                  None, architecture['input_dimension'])

        pricing_model.load_best_model(best_model_filename(i, N, suffix), scaler_filename(i, N, suffix))
        pricing_model.model.eval()

        with torch.no_grad():
            model_pred = pricing_model.transform(torch.Tensor(model_inputs)).cpu()[:, 0]
            predictions_dict[i] = model_pred

    return predictions_dict


def test_statistical_model(features, N, best_model_filename, model_str, suffix='', num_runs=5,
                           splicing_dict={'5': [(0, 2), (8, 10), (16, 17)], '10': [(0, 4), (8, 12), (16, 18)],
                                          '20': [(0, 20)]}):

    """
    Test Ridge model or LightGBM model.
    :param features: List of features to be used as input to the neural network.
    :param N: Number of base processes.
    :param best_model_filename: Filename corresponding to the best model.
    :param model_str: String describing model.
    :param suffix: Suffix string which describes the model. Used to load the best model. Default is ''.
    :param num_runs: Number of independent models over which to perform the test. Default is 5.
    :param splicing_dict: Dictionary describing splicing procedure. Each key contains a list of tuples
                          with each tuple containing the start index and end index used to splice the array.
    :return: Dictionary of predictions. Keys correspond to model number and values are the model predictions.
    """

    predictions_dict = {}
    model_inputs = splice_features(features, N, splicing_dict)
    for i in range(num_runs):
        if 'lgbm' in model_str:

            model = load_lgbm_model(
                load_path_params(best_model_filename(i, N, suffix, model_str))['model'].replace("/LGBM", ""), N)
        else:
            model = load_path_params(best_model_filename(i, N, suffix, model_str))['model']
        predictions_dict[i] = model.predict(model_inputs)

    return predictions_dict


def test_models(targets, speedrs_features, rbf_features, matern_features, model_specs_dict, device,
                num_models=15, suffix='', num_runs=5,
                splicing_dict={'5': [(0, 2), (8, 10), (16, 17)], '10': [(0, 4), (8, 12), (16, 18)], '20': [(0, 20)]}):

    """
    Test all types of models, i.e. test neural networks, ridge models, and LightGBM models.
    :param targets: Target values.
    :param speedrs_features: Input features for SPEEDRS model.
    :param rbf_features: Input features for kernel-based baseline with kernel being the RBF kernel.
    :param matern_features: Input features for kernel-based baseline with kernel being the Matern kernel.
    :param model_specs_dict: Dictionary containing all model specifications, i.e. architecture, filenames, etc...
    :param device: Device on which to run the models.
    :param num_models: Number of models to test. Default is 15.
    :param suffix: Suffix string which describes the model. Used to load the best model. Default is ''.
    :param num_runs: Number of independent models over which to perform the test. Default is 5.
    :param splicing_dict: Dictionary describing splicing procedure. Each key contains a list of tuples
                          with each tuple containing the start index and end index used to splice the array.
    :return: Dictionary containing test results.
    """

    test_results_dict = defaultdict(list)

    for i in tqdm(range(num_models)):

        baseline = model_specs_dict['Baseline'][i]

        if baseline:
            if model_specs_dict['BaselineType'][i] == 'RBF':
                features = rbf_features
            else:
                features = matern_features
        else:
            features = speedrs_features

        model_type = model_specs_dict['ModelType'][i]
        N = model_specs_dict['N'][i]
        if 'NN' in model_type:
            predictions_dict = test_neural_net(model_specs_dict['Architecture'][i],
                                               model_specs_dict['BestModelFilename'][f'{model_type}'],
                                               model_specs_dict['ScalarFilename'][f'{model_type}'],
                                               features, N, device, suffix=suffix, splicing_dict=splicing_dict)
        else:
            if baseline:
                if 'RBF' in model_type:
                    filename_key = 'RBF (Baseline)'
                else:
                    filename_key = 'Matern (Baseline)'
            else:
                filename_key = 'SPEEDRS (Baseline)'

            try:
                del model_specs_dict['BestParamComb'][model_type][N]['MSE']
            except:
                pass

            model_str = model_specs_dict['ModelStr'][model_type](**model_specs_dict['BestParamComb'][model_type][N])
            predictions_dict = test_statistical_model(features, N,
                                                      model_specs_dict['BestModelFilename'][filename_key],
                                                      model_str,
                                                      suffix=suffix, splicing_dict=splicing_dict)

        test_mse = []
        for j in range(num_runs):
            test_mse.append(mean_squared_error(predictions_dict[j], targets))

        test_results_dict['ModelType'].append(model_type)
        test_results_dict['N'].append(N)
        test_results_dict['Baseline'].append(model_specs_dict['Baseline'][i])
        test_results_dict['BaselineType'].append(model_specs_dict['BaselineType'][i])
        test_results_dict['Avg Test MSE'].append(np.mean(test_mse))
        test_results_dict['Std Test MSE'].append(np.std(test_mse))

    return test_results_dict


def get_best_model_NN(N, filename, suffix='', num_runs=5, device='cpu'):

    """
    Find the neural network which obtained the lowest validation error.
    :param N: Number of base processes.
    :param filename: Model filename function. This is a function with input being the model number, N and the suffix
                     and returns a string.
    :param suffix: Suffix string which describes the model. Used to load the best model. Default is ''.
    :param num_runs: Number of independent models over which to perform the test. This indicates the number of models to
                     iterate over with the iteration number being the model number used as input in the filename
                     function. Default is 5.
    :param device: Device on which to run the models. Default is 'cpu'.
    :return: Index of best model.
    """

    valid_losses = []
    for i in range(num_runs):
        valid_losses.append(torch.load(filename(i, N, suffix), map_location=device)['valid_loss_2'])
    return np.argmin(valid_losses)

def get_best_model(N, model_str, filename, suffix='', num_runs=5):

    """
    Find the Ridge model or LightGBM model which obtained the lowest validation error.
    :param N: Number of base processes.
    :param model_str: String describing model.
    :param filename: Model filename function. This is a function with input being the model number, N, the suffix, and
                     the model_str and returns a string.
    :param suffix: Suffix string which describes the model. Used to load the best model. Default is ''.
    :param num_runs: Number of independent models over which to perform the test. This indicates the number of models to
                     iterate over with the iteration number being the model number used as input in the filename
                     function. Default is 5.
    :return: Index of best model.
    """

    valid_losses = []
    for i in range(num_runs):
        valid_losses.append(load_path_params(filename(i, N, suffix, model_str))['valid_mse'])
    return np.argmin(valid_losses)
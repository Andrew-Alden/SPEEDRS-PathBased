import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
from src.RegressionModel.regression_scaler_model import RegressionModelWithScaler
from torch import nn
import torch.cuda
from sklearn.metrics import mean_squared_error


from src.utils import save_path_params, load_path_params



def splice_features(features, N, splicing_dict={'5': [(0, 2), (8, 10), (16, 17)],
                                                '10': [(0, 4), (8, 12), (16, 18)],
                                                '20': [(0, 20)]}):

    """
    Splice features.
    :param features: List of features.
    :param N: Number of base processes.
    :pram splicing_dict: Dictionary describing splicing procedure. Each key contains a list of tuples
                         with each tuple containing the start index and end index used to splice the array.
    :return: Features corresponding to N base processes.
    """

    index_options = splicing_dict[f'{N}']
    indexing_list = []
    for o in index_options:
        indexing_list += list(range(o[0], o[1]))

    return np.asarray(features)[:, indexing_list]


def train_linear_models(Y, features, N, model, model_str, num_runs=5,
                        baseline=False, folder='Models', suffix=''):

    """
    Train Ridge model and LightGBM model.
    :param Y: Targets.
    :param features: Model input features. Array of shape [Number of samples, Input dimension].
    :param N: Number of base processes.
    :param model_str: String describing model.
    :param num_runs: Number of independent training runs. Default is 5.
    :param baseline: Boolean indicating whether model is a baseline model. Default is False.
    :param folder: Folder containing saved model. Default is 'Models'.
    :param suffix: String used to describe model and placed at end of filename. Default is ''.
    :returns Nothing.
    """

    for i in range(num_runs):

        X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2)
        reg = model.fit(X_train, y_train)
        model_dict = {
            'model': model,
            'train_rscore': reg.score(X_train, y_train),
            'train_mse': mean_squared_error(model.predict(X_train), y_train),
            'valid_rscore': reg.score(X_test, y_test),
            'valid_mse': mean_squared_error(model.predict(X_test), y_test)
        }

        if baseline:
            if 'lgbm' in model_str:
                model.booster_.save_model(f'MixtureModels/{folder}/Baseline/{model_str}_test_{i + 1}_base_{N}{suffix}_model.txt')
                model_dict['model'] = f'MixtureModels/{folder}/Baseline/{model_str}_test_{i + 1}_base_{N}{suffix}_model.txt'
            save_path_params(model_dict,
                             f'MixtureModels/{folder}/Baseline/{model_str}_test_{i + 1}_base_{N}{suffix}')
        else:
            if 'lgbm' in model_str:
                model.booster_.save_model(f'MixtureModels/{folder}/N_{N}/{model_str}_test_{i + 1}_base_{N}{suffix}_model.txt')
                model_dict['model'] = f'MixtureModels/{folder}/N_{N}/{model_str}_test_{i + 1}_base_{N}{suffix}_model.txt'
            save_path_params(model_dict,
                             f'MixtureModels/{folder}/N_{N}/{model_str}_test_{i + 1}_base_{N}{suffix}')


def get_stats(N, model_str, num_runs=5, baseline=False, folder='Models', suffix='', verbose=True):

    """
    Compute training and validation statistics for Ridge model and LightGBM model.
    :param N: Number of base processes.
    :param model_str: String describing model.
    :param num_runs: Number of independent training runs. Default is 5.
    :param baseline: Boolean indicating whether model is a baseline model. Default is False.
    :param folder: Folder containing saved model. Default is 'Models'.
    :param suffix: String used to describe model and placed at end of filename. Default is ''.
    :param verbose: Boolean indicating whether to display results. Default is True.
    :return: Average training MSE, Std training MSE, Average Validation MSE, Std Validation MSE.
    """

    train_mse = []
    valid_mse = []

    for i in range(num_runs):

        if baseline:
            model_dict = load_path_params(
                f'MixtureModels/{folder}/Baseline/{model_str}_test_{i + 1}_base_{N}{suffix}')
        else:
            model_dict = load_path_params(
                f'MixtureModels/{folder}/N_{N}/{model_str}_test_{i + 1}_base_{N}{suffix}')
        train_mse.append(model_dict['train_mse'])
        valid_mse.append(model_dict['valid_mse'])

    if verbose:
        print(f'{N}: Train Mean: {np.mean(train_mse)}, Train Std: {np.std(train_mse)}')
        print(f'{N}: Valid Mean: {np.mean(valid_mse)}, Valid Std: {np.std(valid_mse)}')

    return np.mean(train_mse), np.std(train_mse), np.mean(valid_mse), np.std(valid_mse)


def display_linear_model_results(lrs, L2_params, model_str1, model_str2, suffix='', folder='Models', baseline=False,
                                 N_list=[5, 10, 20], display=True):

    """
    Obtain summmary of model performance for Ridge model and LightGBM model.
    :param lrs: List of learning rates.
    :param L2_params: List of regularisation values.
    :param model_str1: String describing ridge model.
    :param model_str2: String describing LightGBM model.
    :param suffix: Filename suffix. Default is ''.
    :param folder: Folder containing saved model. Default is 'Models'.
    :param baseline: Boolean indicating whether model is a baseline model. Default is False.
    :param N_list: List of number of base processes. Default is [5, 10, 20].
    :param display: Boolean indicating whether to display performance results. Default is True.
    :return: Two dictionaries. First dict contains best ridge model and second dict contains best LightGBM model.
             The dictionary keys are the number of base processes and the values are themselves dictionaries containing
             the parameter values corresponding to the best model as well as the validation MSE.
    """


    current_best_ridge = {5: {'MSE': -1, 'l2': -1}, 10: {'MSE': -1, 'l2': -1}, 20: {'MSE': -1, 'l2': -1}}
    current_best_lgbm = {5: {'MSE': -1, 'l2': -1, 'lr': -1}, 10: {'MSE': -1, 'l2': -1, 'lr': -1},
                                 20: {'MSE': -1, 'l2': -1, 'lr': -1}}

    for N in N_list:
        linear_results_dict = defaultdict(list)
        for l2 in L2_params:
            linear_results_dict['Model'].append('Ridge')
            linear_results_dict['N'].append(N)
            linear_results_dict['L2'].append(l2)
            linear_results_dict['LR'].append('N/A')
            model_str = model_str1(l2)
            train_mse, train_std, valid_mse, valid_std = get_stats(N, model_str, suffix=suffix, verbose=False,
                                                                   folder=folder, baseline=baseline)
            linear_results_dict['Train MSE'].append(train_mse)
            linear_results_dict['Train Std'].append(train_std)
            linear_results_dict['Valid MSE'].append(valid_mse)
            linear_results_dict['Valid Std'].append(valid_std)

            if current_best_ridge[N]['MSE'] < 0:
                current_best_ridge[N]['l2'] = l2
                current_best_ridge[N]['MSE'] = valid_mse
            else:
                if valid_mse < current_best_ridge[N]['MSE']:
                    current_best_ridge[N]['l2'] = l2
                    current_best_ridge[N]['MSE'] = valid_mse

        for l2 in L2_params:
            for lr in lrs:
                model_str = model_str2(l2, lr)
                linear_results_dict['Model'].append('LGBM')
                linear_results_dict['N'].append(N)
                linear_results_dict['L2'].append(l2)
                linear_results_dict['LR'].append(lr)
                train_mse, train_std, valid_mse, valid_std = get_stats(N, model_str, suffix=suffix, verbose=False,
                                                                       folder=folder, baseline=baseline)
                linear_results_dict['Train MSE'].append(train_mse)
                linear_results_dict['Train Std'].append(train_std)
                linear_results_dict['Valid MSE'].append(valid_mse)
                linear_results_dict['Valid Std'].append(valid_std)

                if current_best_lgbm[N]['MSE'] < 0:
                    current_best_lgbm[N]['l2'] = l2
                    current_best_lgbm[N]['lr'] = lr
                    current_best_lgbm[N]['MSE'] = valid_mse
                else:
                    if valid_mse < current_best_lgbm[N]['MSE']:
                        current_best_lgbm[N]['l2'] = l2
                        current_best_lgbm[N]['lr'] = lr
                        current_best_lgbm[N]['MSE'] = valid_mse

        linear_results_df = pd.DataFrame(linear_results_dict)
        if display:
            print(linear_results_df)
            print(f'{"*" * 100}')
            print(f'{"*" * 100}')
            print(f'{"*" * 100}')

    return current_best_ridge, current_best_lgbm


def train_models(Y, features, barrier_training_param_dict,
                 barrier_dataset_loader_params, barrier_input_dimension,
                 barrier_model_param_dict, N, device, scheduler_gamma=0.75,
                 use_scheduler=True,
                 milestones=[50, 75, 100, 125, 150, 175, 185, 195, 200, 210, 220, 230, 240, 250, 275, 285, 295],
                 loss=nn.MSELoss(),
                 num_runs=5, baseline=False, baseline_str=None, suffix='', folder='Models', model_desc_string=''):
    """
    Train the neural networks.
    :param Y: Targets. Array of shape [Number of samples].
    :param features: Model input features. Array of shape [Number of samples, Input dimension].
    :param barrier_training_param_dict: Dictionary specifying the model training procedure. The keys are:
                                           -> lr - learning rate.
                                           -> Epochs - number of epochs.
                                           -> l2_weight - L2-regularisation weight.
                                           -> l1_weight - L1-regularisation weight.
                                           -> Train/Val Split - Percentage of data used for training as opposed to validation.
    :param barrier_dataset_loader_params: Dictionary specifying the loading of the dataset. The keys are:
                                             -> batch_size - the batch size.
                                             -> shuffle - Boolean indicating whether to shuffle the order when loading the data.
                                             -> num_workers - specify how many processes are simultaneously loading the data.
                                                              If num_workers=0, the main process loads the data.
    :param barrier_input_dimension: Input dimension.
    :param barrier_model_param_dict: Dictionary specifying the regression model architecture. The keys are:
                                        -> input_dimension - input dimension of the model.
                                        -> intermediate_dimensions - list of hidden layer dimensions.
                                        -> activation_functions - list of activation functions.
                                        -> add_layer_norm - list of Booleans indicating whether to add layer normalisation
                                           before a neural network layer.
                                        -> output_dimension - output dimension of the model.
                                        -> output_activation_fn - output layer activation function.
    :param N: Number of base processes.
    :param device: System device.
    :param scheduler_gamma: scheduler decay factor. Default is 0.75.
    :param use_scheduler: Boolean indicating whether to use a learning rate scheduler. Default is True.
    :param milestones: List of milestones for the MultiStepLR scheduler. Default is [50, 75, 100, 125, 150, 175, 185, 195].
    :param loss: PyTorch or custom loss function. Default is MSE loss.
    :param num_runs: Number of independent training runs. Default is 5.
    :param folder: Root folder. Default is 'Models'.
    :param model_desc_string: String describing model. Default is ''.
    :return: Nothing.
    """

    for i in range(num_runs):

        if baseline:

            pricing_model = RegressionModelWithScaler(barrier_model_param_dict, barrier_training_param_dict,
                                                      barrier_dataset_loader_params, loss, device,
                                                      f'MixtureModels/BarrierOptions/{folder}/Baseline/{baseline_str}_{model_desc_string}_baseline_scaler_test_{i + 1}_base_{N}{suffix}.pkl',
                                                      barrier_input_dimension, scheduler_gamma=scheduler_gamma,
                                                      use_scheduler=use_scheduler, milestones=milestones)

            pricing_model.fit(torch.tensor(np.multiply(1, np.asarray(features))).float(),
                              torch.tensor(np.multiply(1, Y)).float(),
                              **{'filename':
                                     f'MixtureModels/BarrierOptions/{folder}/Baseline/{baseline_str}_{model_desc_string}_baseline_model_checkpoint_test_{i + 1}_base_{N}{suffix}.pth.tar',
                                 'best_model_filename':
                                     f'MixtureModels/BarrierOptions/{folder}/Baseline/{baseline_str}_{model_desc_string}_baseline_model_best_test_{i + 1}_base_{N}{suffix}.pth.tar'})


        else:

            pricing_model = RegressionModelWithScaler(barrier_model_param_dict, barrier_training_param_dict,
                                                      barrier_dataset_loader_params, loss, device,
                                                      f'MixtureModels/{folder}/N_{N}/{model_desc_string}_scaler_test_{i + 1}_base_{N}{suffix}.pkl',
                                                      barrier_input_dimension, scheduler_gamma=scheduler_gamma,
                                                      use_scheduler=use_scheduler, milestones=milestones)

            pricing_model.fit(torch.tensor(np.multiply(1, np.asarray(features))).float(),
                              torch.tensor(np.multiply(1, Y)).float(),
                              **{'filename':
                                     f'MixtureModels/{folder}/N_{N}/{model_desc_string}_model_checkpoint_test_{i + 1}_base_{N}{suffix}.pth.tar',
                                 'best_model_filename':
                                     f'MixtureModels/{folder}/N_{N}/{model_desc_string}_model_best_test_{i + 1}_base_{N}{suffix}.pth.tar'})


def get_stats_NN(N, device='cpu', num_runs=5, baseline=False, baseline_str=None, suffix='', verbose=True,
                 folder='Models', model_desc_string=''):
    """
    Compute training and validation statistics for neural networks.
    :param N: Number of base processes.
    :param num_runs: Number of independent training runs. Default is 5.
    :param folder. Root folder. Default is 'Models'.
    :param model_desc_string: String describing model. Default is ''.
    :return: Nothing.
    """

    train_losses = []
    valid_losses = []

    for i in range(num_runs):

        if baseline:
            path = f'MixtureModels/{folder}/Baseline/{baseline_str}_{model_desc_string}_baseline_model_best_test_{i + 1}_base_{N}{suffix}.pth.tar'

        else:
            path = f'MixtureModels/{folder}/N_{N}/{model_desc_string}_model_best_test_{i + 1}_base_{N}{suffix}.pth.tar'

        checkpoint = torch.load(path, map_location=device)

        train_losses.append(checkpoint['train_loss'])
        valid_losses.append(checkpoint['valid_loss'])

    if verbose:
        print(f'{N}: Train Mean: {np.mean(train_losses)}, Train Std: {np.std(train_losses)}')
        print(f'{N}: Valid Mean: {np.mean(valid_losses)}, Valid Std: {np.std(valid_losses)}')

    return np.mean(train_losses), np.std(train_losses), np.mean(valid_losses), np.std(valid_losses)
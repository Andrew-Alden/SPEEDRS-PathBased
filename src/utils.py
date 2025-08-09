import torch
import pickle
import joblib
import numpy as np

# ===========================================================================================================
# Various utility functions to save and load the datasets.
# ===========================================================================================================

def save_dataset(features, labels, features_file_name, labels_file_name):

    """
    Save dataset.
    :param features: Feature vectors.
    :param labels: Target labels.
    :param features_file_name: Features file name.
    :param labels_file_name: Labels file name.
    :return: Nothing
    """

    torch.save(features, f'{features_file_name}.pt')
    torch.save(labels, f'{labels_file_name}.pt')


def load_dataset(features_file_name, labels_file_name):

    """
    Load the dataset.
    :param features_file_name: Features file name.
    :param labels_file_name: Labels file name.
    :return: 1) Labels.
             2) Features.
    """

    try:
        return torch.load(f'{labels_file_name}.pt'), torch.load(f'{features_file_name}.pt', weights_only=False)
    except Exception as e:
        try:
            return torch.load(f'{labels_file_name}.pt'), torch.load(f'{features_file_name}.pt')
        except Exception as e:
            print(f"❌ Failed: {e}")


def save_path_params(params_dict, file_name):

    """
    Save dictionary.
    :param params_dict: Parameter dictionary.
    :param file_name: File name.
    :return: Nothing
    """

    with open(file_name, 'wb') as f:
        pickle.dump(params_dict, f)


def load_path_params(file_name):

    """
    Load dictionary.
    :param file_name: File name.
    :return: Parameter Dictionary.
    """

    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        try:
            return np.load(f'{file_name}.npy', allow_pickle=True)
        except Exception as e:
            print(f"❌ Failed: {e}")
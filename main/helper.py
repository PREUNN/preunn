from models import abstract_architecture
import torch
import os


def load_model(model_save_path: str, architecture: abstract_architecture):
    """
    Method to load or create a model
    :param model_save_path: path to model
    :param architecture: type of network to be loaded
    :return: loaded or created model
    """
    if os.path.isfile(model_save_path):
        model = torch.load(model_save_path)
        print("loaded ", model)
    else:
        model = architecture()
        print("new model created")
    return model


def load_data(data_save_path: str):
    """
    Simple loading routine for a custom dataset
    :param data_save_path: path of dataset
    :return: returns loaded dataset or false
    """
    if os.path.isfile(data_save_path):
        dataset = torch.load(data_save_path)
        print("loaded " + data_save_path)
        return dataset
    else:
        return False


def save_dataset(dataset, path: str):
    """
    Simple saving routine
    :param dataset: dataset to save
    :param path: path where to save
    :return:
    """
    torch.save(dataset, path)
    print("Saved dataset: ", dataset)

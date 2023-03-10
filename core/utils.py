import importlib
import datetime
import os
from stable_baselines3.common.base_class import BaseAlgorithm
from omegaconf import DictConfig, OmegaConf
import time

time_format = '%m-%d_%Hh%Mmin'


def string_to_class(class_string : str):
    """Get a class from a string of the form "module_name:class_name"

    Args:
        class_string (str): a string of the form "module_name.file_name:class_name"

    Returns:
        Type: the class
    """
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def time_to_str(time_instant : datetime.datetime) -> str:
    return time_instant.strftime(time_format)

def str_to_time(time_string : str) -> datetime.datetime:
    return datetime.datetime.strptime(time_string, time_format)


def create_model_path(
        model_dir : str, 
        algo_name : str,
        env_name : str,
        timesteps : int, 
        reward : float):
    """Generate a model name from the model directory, the run config and the current timestep
    and mean reward of the model.

    Args:
        model_dir (str): the path to the folder where the model will be saved
        cfg (dict): the run config
        timesteps (int): the timestep (number of steps) of the model
        reward (float): the mean reward of this model

    Returns:
        str: the model name
    """
    model_name = model_dir + '/'
    model_name += algo_name + ' '
    model_name += env_name + ' '
    model_name += time_to_str(datetime.datetime.now()) + ' '
    model_name += f"t={timesteps}" + ' '
    model_name += f"r={reward:.2f}" + ' '
    model_name = model_name[:-1]
    return model_name

class InfoModel:
    """A data class giving information about a model.
    """
    def __init__(self, model_path : str) -> None:
        self.model_path = model_path
        self.model_name = self.model_path.split('/')[-1]
        self.algo_name, self.env_name, self.time, self.timesteps, self.reward = self.model_name.split(' ')

def choose_model_to_load(
        models_path : str, 
        env_name : str = None,
        algo_name : str = None,
        criteria : str = 'reward',
        ) -> str:
    """Choose a model path according to a certain criteria (default reward).
    It oly searches among the models in the folder models_path with the corresponding env and algorithm.

    Args:
        models_path (str): the path to the folder containing the models
        env_name (str, optional): the name of the environment. Defaults to None (no constraints).
        algo_name (str, optional): the name of the algorithm. Defaults to None (no constraints).
        criteria (str, optional): the criteria for choosing the model. Defaults to 'r'.

    Returns:
        str: the best model path
    """
    # Get models as InfoModel objects
    models = os.listdir(models_path)
    models = [InfoModel(models_path + '/' + model) for model in models]
    # Filter models
    if env_name is not None:
        models = [model for model in models if model.env_name == env_name]
    if algo_name is not None:
        models = [model for model in models if model.algo_name == algo_name]
    # Select best model
    if criteria == 'reward':
        models = sorted(models, key=lambda model: model.reward, reverse=True)
    elif criteria == 'timesteps':
        models = sorted(models, key=lambda model: model.timesteps, reverse=True)
    elif criteria == 'time':
        models = sorted(models, key=lambda model: str_to_time(model.time), reverse=True)
    else:
        raise ValueError(f"criteria {criteria} not understood, choose between 'reward', 'timesteps' and 'time'")
    return models[0].model_path


def try_to_load(
        model : BaseAlgorithm,
        algo_name : str,
        env_name : str,
        checkpoint : str,
        criteria : str,
        ):
    """Try to load a BaseAlgorithm SB3 model from a checkpoint directory or file.

    Args:
        model (BaseAlgorithm): the model that requires loading
        checkpoint (str): the path to the checkpoint directory or file
        cfg (DictConfig): the run config

    Returns:
        BaseAlgorithm: the model with the loaded parameters if possible, else the same model
    """
    print(f"Loading checkpoint from {checkpoint} with criteria {criteria}...")
    if checkpoint is not None:
        # If dir, search model
        if os.path.isdir(checkpoint):
            print(f"\nPicking model to load from {checkpoint}...")
            try: 
                model_path = choose_model_to_load(
                    models_path=checkpoint,
                    env_name=env_name,
                    algo_name=algo_name,
                    criteria=criteria,
                    )
                model.load(model_path)
                print(f"Model successfully loaded : {model_path}")
            except Exception as e:
                print(f"WARNING : Model loading failed : {e}\n -> Training from scratch")
        # If file, load model
        elif os.path.isfile(checkpoint):
            try:
                model.load(checkpoint)
                print(f"Model successfully loaded : {checkpoint}")
            except Exception as e:
                print(f"WARNING : Model loading failed : {e}\n -> Training from scratch")
        else:
            print(f"WARNING : {checkpoint} is neither a file nor a directory\n -> Training from scratch")
    else:
        print("No checkpoint to load -> Training from scratch")
    return model
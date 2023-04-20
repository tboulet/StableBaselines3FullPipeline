import importlib
import datetime
import os
from typing import Any, Dict, List, Type, Union
import gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from omegaconf import DictConfig, OmegaConf
import time

time_format = '%m-%d_%Hh%Mmin'


def string_to_class(class_string : str) -> Type:
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

def try_get(cfg : OmegaConf, key : str, default : Any = None) -> Any:
    """Try to get a value from a config. If the key is not found, return the default value.

    Args:
        cfg (OmegaConf): the config
        key (str): the key to look for
        default (Any, optional): the default value to return if the key is not found. Defaults to None.

    Returns:
        Any: the value found in the config or the default value
    """
    if key not in cfg:
        return default
    return cfg[key]
    
def try_get_dict(cfg : OmegaConf, key : str, default : dict = {}) -> dict:
    """Try to get a dict from a config. If the key is not found, return an empty dict.

    Args:
        cfg (OmegaConf): the config
        key (str): the key to look for
        default (dict, optional): the default value to return if the key is not found. Defaults to {}.

    Returns:
        dict: the dict found in the config or the default value
    """
    if key not in cfg or cfg[key] is None:
        return default
    return cfg[key]

def try_get_list(cfg : OmegaConf, key : str, default : list = []) -> list:
    """Try to get a list from a config. If the key is not found, return an empty list.

    Args:
        cfg (OmegaConf): the config
        key (str): the key to look for
        default (list, optional): the default value to return if the key is not found. Defaults to [].

    Returns:
        list: the list found in the config or the default value
    """
    if key not in cfg or cfg[key] is None:
        return default
    return cfg[key]
    
def none_to_infs(*args):
    """Replace None values by np.inf.
    """
    return [np.inf if arg is None else arg for arg in args]

def none_to_empty_dict(*args):
    """Replace None values by empty dict.
    """
    return [{} if arg is None else arg for arg in args]

def none_to_empty_list(*args):
    """Replace None values by empty list.
    """
    return [[] if arg is None else arg for arg in args]

def create_run_name(
        algo_name : str, 
        env_name : str, 
        ) -> str:
    """Generate a run name from the run config and the current timestep and mean reward of the model.

    Args:
        algo_name (str): the name of the algorithm
        env_name (str): the name of the environment
        time_instant (datetime.datetime, optional): the time instant. Defaults to None (current time).

    Returns:
        str: the run name
    """
    run_name = algo_name + '_'
    run_name += env_name + '_'
    run_name += time_to_str(datetime.datetime.now())
    return run_name

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
        algo_name (str): the name of the algorithm
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


def extract_class_if_class_string(class_string_or_name : str) -> Union[str, Type]:
    """Extract the class name from an object or a string.

    Args:
        class_string_or_name (str): the class as a string or simply a string

    Returns:
        str: the class name
    """
    if ":" in class_string_or_name:
        return string_to_class(class_string_or_name)
    else:
        return class_string_or_name

def try_to_seed(env, seed : int = None):
    """Try to seed the environment with the given seed. If the env does not support seeding, raise a warning.

    Args:
        env (gym.Env): the environment
        seed (int, optional): the seed. Defaults to None.

    Returns:
        gym.Env: the environment with the seed applied (or not)
    """
    if seed is not None:
        try:
            env.seed(seed)
        except Exception as e:
            print(f"WARNING : Environment {env} seeding failed : {e}")
    return env


def class_string_to_class(obj : Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """Recursively turn a nested object with class strings into a nested object with classes.
    For every object inside obj, each string module.file:Class will be turned into the class Class from the module file in the module module.

    Args:
        obj (Union[Dict, List, Any]): the obj to transform

    Returns:
        Union[Dict, List, Any]: the transformed obj
    """
    if isinstance(obj, dict):
        return {key: class_string_to_class(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [class_string_to_class(value) for value in obj]
    elif isinstance(obj, str):
        return extract_class_if_class_string(obj)
    else:
        return obj


def replace_by_class_if_class_string(cfg : DictConfig, key : str) -> None:
    """Replace a key in a config by its corresponding class if it is a class string.

    Args:
        cfg (DictConfig): the config
        key (str): the key to replace
    """
    if key in cfg:
        obj = cfg[key]
        cfg[key] = extract_class_if_class_string(obj)
# Env
import gym
from stable_baselines3.common.env_checker import check_env
# Agent
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
# Vectorized env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
# Evaluation
from stable_baselines3.common.evaluation import evaluate_policy
import torch
# Callbacks
from core.callback_sb3 import CustomEvalCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
# Seeding
from stable_baselines3.common.utils import set_random_seed
# Utils
from core.utils import (
    try_get_dict, 
    none_to_empty_dict, 
    none_to_empty_list, 
    none_to_infs, 
    string_to_class, 
    create_run_name,
    create_model_path, 
    try_get_list, 
    try_get, 
    try_to_load,
    try_to_seed,
    extract_class_if_class_string,
    replace_by_class_if_class_string,
    class_string_to_class,
)

# Config management
import hydra
from omegaconf import DictConfig, OmegaConf
import pprint 
# Other
import numpy as np
from typing import Any, Callable, Type, Union, List

pp = pprint.PrettyPrinter(indent=4)



@hydra.main(version_base=None, config_path="configs", config_name="enjoy_config")
def main(cfg : DictConfig):

    # Convert OmegaConf to dict and convert all class strings to classes
    cfg = OmegaConf.to_container(cfg)
    cfg = class_string_to_class(cfg)
    print("Enjoying with config :")
    pp.pprint(cfg)
    
    # Eval numerical parameters
    n_enjoy_episodes : int = cfg['training']['n_enjoy_episodes']
    n_eval_episodes : int = cfg['training']['n_eval_episodes']
    
    # Model
    AlgoClass : Type[BaseAlgorithm] = cfg['algo']['class']
    algo_name = try_get(cfg['algo'], "name", default = AlgoClass.__name__)
    algo_cfg = try_get_dict(cfg['algo'], "algo_cfg")
    env_wrappers_from_algo : List[dict] = try_get_list(cfg['algo'], "env_wrappers")
    
    PolicyClass : Union[str, Type[BasePolicy]] = cfg['policy']['class']
    policy_cfg  = try_get_dict(cfg['policy'], "policy_cfg")
    
    # Environment
    create_env_fn : Callable[..., gym.Env] = cfg['env']['class']
    env_cfg : dict = try_get_dict(cfg['env'], "env_cfg")
    env_wrappers : List[dict] = try_get_list(cfg['env'], "env_wrappers")
    
    checkpoint : str = cfg['training']['checkpoint']
    checkpoint_criteria : str = cfg['training']['checkpoint_criteria']

    # Logging
    verbose : int = cfg['training']['verbose']
    log_path : str = cfg['training']['log_path']

    # Names
    project_name : str = cfg['training']['project_name']
    env_name : str = cfg['env']['name']
    algo_name : str = cfg['algo']['name']
    
    # Seeding
    seed : int = cfg['training']['seed']
    if seed is None: seed = np.random.randint(0, 2**32 - 1)
    set_random_seed(seed, using_cuda=True)

    # Rewriting some config parameters
    env_cfg, algo_cfg = none_to_empty_dict(env_cfg, algo_cfg)
    env_wrappers, env_wrappers_from_algo = none_to_empty_list(env_wrappers, env_wrappers_from_algo)
    
    # Environment
    print("Using gym environment")
    env = create_env_fn(**env_cfg)
    env = try_to_seed(env, seed)
    env_monitored = Monitor(env, log_path)
    env = env_monitored
    for wrapper_info_dict in env_wrappers + env_wrappers_from_algo:
        if wrapper_info_dict["class"] is not None:
            wrapper_args, = none_to_empty_dict(wrapper_info_dict["args"])
            wrapper_class = wrapper_info_dict["class"]
            env = wrapper_class(env, **wrapper_args)
    try:
        check_env(env, warn=True)
    except Exception as e:
        print(f"Warning : check_env failed with error : {e}")
    
    # Instantiate the agent
    model : BaseAlgorithm = AlgoClass(
        policy = PolicyClass,
        policy_kwargs = policy_cfg, 
        env = env, 
        verbose = verbose,
        seed = seed,
        )
    print(f"Model policy: {model.policy}")
    model = try_to_load(
        model=model,
        algo_name=algo_name,
        env_name=env_name,
        checkpoint=checkpoint,
        criteria=checkpoint_criteria,
        )
        
    # Evaluate the agent
    print("\nEvaluating...")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Enjoy trained agent
    for episode in range(n_enjoy_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
    
    
    

if __name__ == "__main__":
    main()
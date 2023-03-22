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


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg : DictConfig):
    # Convert OmegaConf to dict and convert all class strings to classes
    cfg = OmegaConf.to_container(cfg)
    cfg = class_string_to_class(cfg)
    print("Training with config :")
    pp.pprint(cfg)
    
    # Training & eval numerical parameters
    timesteps : int = cfg['training']['timesteps']
    eval_freq : int = cfg['training']['eval_freq']
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
    n_envs : dict = try_get(cfg['training'], "n_envs", default = 0)
    env_wrappers : List[dict] = try_get_list(cfg['env'], "env_wrappers")
    
    # Loading 
    checkpoint : str = cfg['training']['checkpoint']
    checkpoint_criteria : str = cfg['training']['checkpoint_criteria']

    # Logging
    verbose : int = cfg['training']['verbose']
    do_wandb : bool = cfg['training']['do_wandb']
    do_tensorboard : bool = cfg['training']['do_tensorboard']
    
    # Logging directories
    log_path : str = cfg['training']['log_path']
    log_path_tb : str = cfg['training']['log_path_tb'] if do_tensorboard else None
    models_path : str = cfg['training']['models_path']
    best_model_path : str = cfg['training']['best_model_path']
    final_model_path : str = cfg['training']['final_model_path']

    # Names
    project_name : str = cfg['training']['project_name']
    env_name : str = cfg['env']['name']
    algo_name : str = cfg['algo']['name']

    # Seeding
    seed : int = cfg['training']['seed']
    if seed is None: seed = np.random.randint(0, 2**32 - 1)
    set_random_seed(seed, using_cuda=True)

    # Rewriting some config parameters
    timesteps, eval_freq = none_to_infs(timesteps, eval_freq)
    env_cfg, algo_cfg = none_to_empty_dict(env_cfg, algo_cfg)
    env_wrappers, env_wrappers_from_algo = none_to_empty_list(env_wrappers, env_wrappers_from_algo)
    
    # Environment
    if n_envs == 0:
        raise ValueError("n_envs must be at least 1")
    elif n_envs == 1:
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
    else:
        print(f"Using {n_envs} gym vectorized environments")
        def make_env(rank:int, **env_cfg):
            env = create_env_fn(**env_cfg)
            env = try_to_seed(env, seed+rank)
            return env
        vec_env = DummyVecEnv([lambda:make_env(rank=i, **env_cfg) for i in range(n_envs)])
        monitored_vec_env = VecMonitor(vec_env, log_path)
        env = monitored_vec_env
        for wrapper_info_dict in env_wrappers + env_wrappers_from_algo:
            if wrapper_info_dict["vec_class"] is not None:
                vec_wrapper_args, = none_to_empty_dict(wrapper_info_dict["vec_args"])
                wrapper_class = wrapper_info_dict["vec_class"]
                env = wrapper_class(env, **vec_wrapper_args)
    try:
        check_env(env, warn=True)
    except Exception as e:
        print(f"Warning : check_env failed with error : {e}")
    
    # Callbacks
    callback = []
    eval_cb = CustomEvalCallback(
        algo_name=algo_name,
        env_name=env_name,
        eval_env = env, 
        callback_on_new_best=None,
        callback_after_eval=None,
        eval_freq=eval_freq, 
        n_eval_episodes=n_eval_episodes, 
        log_path=log_path, 
        best_model_save_path=best_model_path,
        deterministic=True, 
        render=False,
        verbose=0,
        )
    callback.append(eval_cb)


    # Start WandB run
    if do_wandb:
        run = wandb.init(
            project=project_name,
            config=dict(),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb_cb = WandbCallback()
        callback.append(wandb_cb)

    # Instantiate the agent
    model : BaseAlgorithm = AlgoClass(
        policy = PolicyClass,
        policy_kwargs = policy_cfg, 
        env = env, 
        verbose = verbose, 
        tensorboard_log = log_path_tb,
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
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_path_tb)
    try:
        writer.add_graph(model = model.policy, input_to_model = torch.tensor(env.reset()))
    except Exception as e:
        print(f"Warning: could not add graph to tensorboard : {e}")
    print(f"Model policy: {model.policy}")
    
    # Train the agent and display a progress bar
    print("\nTraining...")
    model.learn(total_timesteps=int(timesteps), callback=callback, tb_log_name="ppo_harvest_run", reset_num_timesteps=checkpoint is not None)
    print("Training done.")

    # Evaluate the agent
    print("\nEvaluating...")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the agent
    print("\nSaving final model...")
    model_name = create_model_path(
        model_dir = final_model_path,
        algo_name=algo_name,
        env_name=env_name,
        timesteps = timesteps,
        reward = mean_reward,)
    model.save(model_name)
    print(f"Model saved at {model_name}")
    del model
    if do_wandb:
        run.finish()

    
    
    

if __name__ == "__main__":
    main()
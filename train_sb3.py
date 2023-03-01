# Env
from typing import Any, Callable, Type
import gym
# Agent
from stable_baselines3.common.base_class import BaseAlgorithm
# Vectorized env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
# Evaluation
from stable_baselines3.common.evaluation import evaluate_policy
# Callbacks
from core.callback_sb3 import CustomEvalCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
# Seeding
from stable_baselines3.common.utils import set_random_seed
# Utils
from core.utils import string_to_class, create_model_path, try_to_load, try_to_load
# Config management
import hydra
from omegaconf import DictConfig, OmegaConf
# Other
from typing import Any, Callable, Type



@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg : DictConfig):
    print("\nTraining with config :\n", OmegaConf.to_yaml(cfg))
    
    # Training & eval numerical parameters
    timesteps = cfg.training.timesteps
    eval_freq = cfg.training.eval_freq
    n_eval_episodes = cfg.training.n_eval_episodes
    
    # Model
    algo_class_string = cfg.algo.class_string

    # Environment
    create_env_fn_string = cfg.env.create_env_fn_string
    create_env_fn : Callable[..., gym.Env] = string_to_class(create_env_fn_string)
    env_cfg = cfg.env.env_cfg if cfg.env.env_cfg is not None else {}
    n_envs = cfg.training.n_envs

    # Loading 
    checkpoint = cfg.training.checkpoint
    checkpoint_criteria = cfg.training.checkpoint_criteria

    # Logging
    do_wandb = cfg.training.do_wandb

    # Logging directories
    log_path = cfg.training.log_path
    log_path_tb = cfg.training.log_path_tb
    models_path = cfg.training.models_path
    best_model_path = cfg.training.best_model_path
    final_model_path = cfg.training.final_model_path

    # Names
    project_name = cfg.training.project_name
    env_name = cfg.env.name
    algo_name = cfg.algo.name

    # Seeding
    seed = cfg.training.seed
    set_random_seed(seed, using_cuda=True)



    # Environment
    if n_envs == 1:
        print("Using gym environment")
        env = create_env_fn(**env_cfg)
        env_monitored = Monitor(env, log_path)
        env = env_monitored
    else:
        print(f"Using {n_envs} gym vectorized environments")
        vec_env = DummyVecEnv([lambda:create_env_fn(**env_cfg) for i in range(n_envs)])
        monitored_vec_env = VecMonitor(vec_env, log_path)
        env = monitored_vec_env


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
    AlgoClass : Type[BaseAlgorithm] = string_to_class(algo_class_string)
    model : BaseAlgorithm = AlgoClass(
        policy = "MlpPolicy", 
        env = env, 
        verbose = 1, 
        tensorboard_log = log_path_tb,
        seed = seed,
        )
    model = try_to_load(
        model=model,
        algo_name=algo_name,
        env_name=env_name,
        checkpoint=checkpoint,
        criteria=checkpoint_criteria,
        )
    
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
# Env
import gym
# Agent
from stable_baselines3.common.base_class import BaseAlgorithm
# Evaluation
from stable_baselines3.common.evaluation import evaluate_policy
# Seeding
from stable_baselines3.common.utils import set_random_seed
# Utils
from core.utils import string_to_class, create_model_path, try_to_load, try_to_load
# Config management
import hydra
from omegaconf import DictConfig, OmegaConf
# Other
from typing import Any, Callable, Type



@hydra.main(version_base=None, config_path="configs", config_name="enjoy_config")
def main(cfg : DictConfig):
    print("\nEnjoying SB3 agent with config :\n", OmegaConf.to_yaml(cfg))

    # Model
    algo_class_string = cfg.algo.class_string

    # Environment
    create_env_fn_string = cfg.env.create_env_fn_string
    create_env_fn : Callable[..., gym.Env] = string_to_class(create_env_fn_string)
    env_cfg = cfg.env.env_cfg if cfg.env.env_cfg is not None else {}
    env = create_env_fn(**env_cfg)

    # Loading 
    checkpoint = cfg.training.checkpoint
    checkpoint_criteria = cfg.training.checkpoint_criteria

    # Names
    project_name = cfg.training.project_name
    env_name = cfg.env.name
    algo_name = cfg.algo.name

    # Seeding
    seed = cfg.training.seed



    # Instantiate the agent
    AlgoClass : Type[BaseAlgorithm] = string_to_class(algo_class_string)
    model : BaseAlgorithm = AlgoClass(
        policy = "MlpPolicy", 
        env = env, 
        verbose = 1, 
        seed = seed,
        )
    model = try_to_load(
        model=model,
        algo_name=algo_name,
        env_name=env_name,
        checkpoint=checkpoint,
        criteria=checkpoint_criteria,
        )

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Enjoy trained agent
    obs = env.reset()
    for episode in range(5):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()


if __name__ == "__main__":
    main()
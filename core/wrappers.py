# File containing usual wrappers for gym environments, mainly vectorized environments.

# VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Atari
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env as make_atari_vec_env

# Frame stacking
from stable_baselines3.common.vec_env import VecFrameStack

# Normalization
from stable_baselines3.common.vec_env import VecNormalize

# Action masking
from sb3_contrib.common.wrappers import ActionMasker
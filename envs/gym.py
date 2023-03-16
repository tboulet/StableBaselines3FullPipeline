import gym

def make_registered_gym_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    return env
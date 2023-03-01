import gym

def create_env_fn():
    env = gym.make("CartPole-v0")
    return env
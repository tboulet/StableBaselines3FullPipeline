import gym

class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
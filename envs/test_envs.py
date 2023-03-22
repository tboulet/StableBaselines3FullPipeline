# Test environments for the SB3 framework, to verify your networks can handle complex observation/actions spaces.
# We assume every observation space used is a Box space (or has been converted to one).



from typing import Any
import gym
import numpy



class TestEnvObservationDict(gym.Env):
    """Environment where the observation is a dictionary of Box 1-dimensional spaces. To test that the observation space is correctly handled."""
    
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "observation1": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=numpy.uint8),
            "observation2": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=numpy.uint8),
        })
        self.action_space = gym.spaces.Discrete(2)
        super().__init__()
        
    def reset(self) -> Any:
        return self.observation_space.sample()
    
    def step(self, action: Any) -> Any:
        return self.observation_space.sample(), 0, False, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass
    

class TestEnvActionMasking(gym.Env):
    """Environment where the observation is a dictionary of Box 1-dimensional spaces. To test that the observation space is correctly handled.
    Additionally, the info dict contains a mask indicating you can't take certain actions.
    """
    
    def __init__(self) -> None:
        self.n_actions = 4
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=numpy.uint8),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=numpy.uint8),
        })
        self.state = self.observation_space.sample()
        super().__init__()
        
    def reset(self) -> Any:
        self.state = self.observation_space.sample()
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        return self.state
    
    def step(self, action: Any) -> Any:
        if self.state['mask'][action] == 0:
            raise ValueError(f"Action {action} is illegal but was taken.")
        self.state = self.observation_space.sample()
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        return self.state, 0, False, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass    
    
    
class TestEnvNonConstantObservation(gym.Env):
    """An environment where the observation contains a batch of vectors of shape (B, *obs_dim)
    B is not constant between episodes, and even between steps.
    For example you may face this kind of environments when you have to deal with a group of vectors representing a group of ennemies.
    """
    
    def __init__(self) -> None:
        self.observation_space = None # TODO : create a custom space
        self.action_space = gym.spaces.Discrete(2)
        super().__init__()
        
    def reset(self) -> Any:
        B = numpy.random.randint(1, 10)
        return numpy.random.random(size = (B, 4))
    
    def step(self, action: Any) -> Any:
        B = numpy.random.randint(1, 10)
        return numpy.random.random(size = (B, 4)), 0, False, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass
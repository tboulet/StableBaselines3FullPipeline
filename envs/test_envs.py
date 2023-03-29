# Test environments for the SB3 framework, to verify your networks can handle complex observation/actions spaces.
# We assume every observation space used is a Box space (or has been converted to one).



from typing import Any, Dict, Tuple, Union
import gym
import numpy

# Observation space as a dictionary of Box 1-dimensional spaces
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
        done = numpy.random.rand() < 0.1
        return self.observation_space.sample(), 1, done, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass
    
from sb3_contrib import MaskablePPO
# Observation space as a dictionary of Box 1-dimensional spaces, one of which is a mask
class TestEnvActionMasking(gym.Env):
    """Environment where the observation is a dictionary of Box 1-dimensional spaces. To test that the observation space is correctly handled.
    Additionally, the info dict contains a mask indicating you can't take certain actions.
    """
    
    def __init__(self) -> None:
        self.n_actions = 4
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict({
            "observation1": gym.spaces.Box(low=0, high=1, shape=(7,), dtype=numpy.uint8),
            "observation2": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=numpy.uint8),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=numpy.uint8),
        })
        self.state = self.observation_space.sample()
        super().__init__()
        
    def reset(self) -> Any:
        self.state = self.observation_space.sample()
        self.state["mask"] = numpy.random.randint(0, 2, size=self.n_actions)  # random mask
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        return self.state
    
    def step(self, action: Any) -> Any:
        if self.state['mask'][action] == 0:
            raise ValueError(f"Action {action} is illegal with mask {self.state['mask']} but was taken.")
        self.state["mask"] = numpy.random.randint(0, 2, size=self.n_actions)  # random mask
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        done = numpy.random.rand() < 0.1
        return self.state, 1, done, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass    

    def action_masks(self) -> numpy.ndarray:
        return self.state["mask"]
    

# Observation space as a dictionary of Box 1-dimensional spaces, one of which is a mask, the other may have non constant size
class TestEnvNonConstantObservationActionMasking(gym.Env):
    """An environment where the observation contains a dictionnary of batch of vectors of shape (B, *obs_dim)
    B is not constant between episodes, and even between steps.
    For example you may face this kind of environments when you have to deal with a group of vectors representing a group of ennemies.
    """
    
    def __init__(self) -> None:
        self.n_actions = 4
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict({
            "observation1": gym.spaces.Box(low=0, high=1, shape=(10, 1,), dtype=numpy.uint8),
            "observation2": gym.spaces.Box(low=0, high=1, shape=(10, 2,), dtype=numpy.uint8),
            "observation3": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=numpy.uint8),
            "mask": gym.spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=numpy.uint8),
        })

    def reset(self) -> Any:
        self.state = self.observation_space.sample()
        self.state["mask"] = numpy.random.randint(0, 2, size=self.n_actions)  # random mask
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        return self.state
    
    def step(self, action: Any) -> Any:
        if self.state['mask'][action] == 0:
            raise ValueError(f"Action {action} is illegal with mask {self.state['mask']} but was taken.")
        self.state["mask"] = numpy.random.randint(0, 2, size=self.n_actions)  # random mask
        self.state["mask"][0] = 1  # to ensure at least one action is possible
        done = numpy.random.rand() < 0.1
        return self.state, 1, done, {}
    
    def render(self, mode: str = 'human', **kwargs) -> Any:
        pass 


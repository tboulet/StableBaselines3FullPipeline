# Adapting the sb3contrib MaskablePPO to work with this framework without hardcoding it, to avoid the need of the evaluation function of maskable_ppo.
# Any masked env must :
# - have a observation space that is a gym.spaces.Dict, one of the keys of which is "mask"
# - have a action_masks(self) method that returns the mask
# - the observation["mask"] must be the mask
# - the mask is a numpy.ndarray of shape (n_actions,) with 1s where the action is allowed and 0s where it is not

from sb3_contrib import MaskablePPO


class MaskablePPO_Adapted(MaskablePPO):

    def predict(self, observation, state = None, episode_start = None, deterministic: bool = False, action_masks = None):
        if action_masks is None:
            action_masks = observation["mask"]
        return super().predict(observation, state, episode_start, deterministic, action_masks)
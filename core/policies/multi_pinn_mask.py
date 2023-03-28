
    
    
from importlib.metadata import distribution
from pyexpat import features
from turtle import forward
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

class PermutationInvariantOperations(nn.Module):
    def __init__(self, dim=1):
        super(PermutationInvariantOperations, self).__init__()
        self.dim = dim
    
    def forward(self, x):  # (B, N, dim_obs)
        max_vals, _ = th.max(x, dim=self.dim)
        min_vals, _ = th.min(x, dim=self.dim)
        sum_vals = th.sum(x, dim=self.dim)
        avg_vals = th.mean(x, dim=self.dim)
        # output = th.stack([max_vals, min_vals, sum_vals, avg_vals], dim=1) # (B, 4, dim_obs)
        output = th.cat([max_vals, min_vals, sum_vals, avg_vals], dim=1) # (B, 4*dim_obs)
        return output
    

class MultiInputPINNWithMaskExtractor(BaseFeaturesExtractor):
    """The features extractor for the MultiInputWithMask policy.
    The observations are typically a dict of numpy arrays, one of them being the mask with key 'mask' :
    obs = {
        'observation1': np.ndarray, # shape (dim_obs1,) 
        'observation2': np.ndarray, # shape (dim_obs2,)
        ...
        'observationN': np.ndarray, # shape (dim_obsN,)
        'mask': np.ndarray,         # shape (n_actions,)
        }
    """
    def __init__(self, observation_space: spaces.Dict, n_neurons_encoder = 16, n_neurons_batchable_encoder = 16, n_neurons_batchable_hidden = 16):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        if 'mask' not in observation_space.spaces.keys():
            raise ValueError(f"Key {key} not found in observation space {observation_space} for MultiInputWithMaskExtractor. Please add a mask with corresponding key 'mask' to the observation space.")
        super().__init__(observation_space, features_dim=1)
        observation_encoders = {}
        batchable_observation_encoders = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key != "mask":
                # Get the shape of the observation
                shape = subspace.shape
                # Compute the size of the output feature vector
                if len(shape) == 1:
                    n_input = shape[0]
                    observation_encoders[key] = nn.Sequential(
                        nn.Linear(n_input, n_neurons_encoder), nn.ReLU(), 
                        )
                    total_concat_size += n_neurons_encoder
                elif len(shape) == 2:
                    n_input = shape[1]
                    batchable_observation_encoders[key] = nn.Sequential(
                        nn.Linear(n_input, n_neurons_batchable_hidden), nn.ReLU(),
                        nn.Linear(n_neurons_batchable_hidden, n_neurons_batchable_encoder), PermutationInvariantOperations(dim=1),
                        )
                    total_concat_size += 4 * n_neurons_batchable_encoder             
                else:
                    raise ValueError(f"Observation {key} has shape {shape} which is not supported by MultiInputWithMaskExtractor. Please use a shape of (n,) or (m, n).")

        self.encoders = nn.ModuleDict(observation_encoders)
        self.batchable_encoders = nn.ModuleDict(batchable_observation_encoders)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, dict_observations : Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        list_batch_of_encoding = []
        for key, batch_of_partial_observations in dict_observations.items():  # (N, *dim_obs)
            if key == "mask":
                pass
            elif len(batch_of_partial_observations.shape) == 2:
                batch_of_encoding = self.encoders[key](batch_of_partial_observations)  # (N, 64)
                list_batch_of_encoding.append(batch_of_encoding)
            elif len(batch_of_partial_observations.shape) >= 3:
                batch_of_encoding = self.batchable_encoders[key](batch_of_partial_observations)
                list_batch_of_encoding.append(batch_of_encoding)
            else:
                raise ValueError(f"Observation {key} has shape {batch_of_partial_observations.shape} which is not supported by MultiInputWithMaskExtractor. Please use a shape of (n,) or (m, n).")
        batch_of_encoding_concat = th.cat(list_batch_of_encoding, dim=1) # (N, 64 + 64 + ... + 64 = 64 * n_observations)

        return {"features": batch_of_encoding_concat, "mask": dict_observations["mask"]}
    



class ActorAndCriticNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """

    def __init__(
        self,
        feature_dim: int = 60,
        last_layer_dim_pi: int = 61,
        last_layer_dim_vf: int = 62,
    ):
        """Custom network for policy and value function.

        Args:
            feature_dim (int, optional): dimension of the features extracted with the features_extractor (e.g. features from a CNN). Defaults to 60.
            last_layer_dim_pi (int, optional): number of units for the last layer of the policy network. Defaults to 61.
            last_layer_dim_vf (int, optional): number of units for the last layer of the value network. Defaults to 62.
        """
        super().__init__()
        
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """The forward pass of the network.

        Args:
            features (Dict[str, th.Tensor]): the extracted features from the features extractor, as a dict with key 'features' and key 'mask'

        Returns:
            Tuple[th.Tensor, th.Tensor]: the logits and the value estimates
        """
        features_observations = features["features"]
        logits = self.policy_net(features_observations)
        critics = self.value_net(features_observations)        
        return logits, critics

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """The forward pass of the policy network.

        Args:
            features (th.Tensor): the extracted features from the features extractor

        Returns:
            th.Tensor: the logits
        """
        features_observations = features["features"]
        return self.policy_net(features_observations)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """The forward pass of the critic network.

        Args:
            features (th.Tensor): the extracted features from the features extractor

        Returns:
            th.Tensor: the value estimates
        """
        features_observations = features["features"]
        return self.value_net(features_observations)






class MultiInputPINNWithMaskPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorAndCriticNetwork(self.features_dim)

    def forward(self, observations: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        # We do not use the base class forward method
        # as we need to pass the mask to the network
        # Preprocess the observation if needed
        features = self.extract_features(observations)
        latent_vf = self.mlp_extractor.forward_critic(features) # (N, 64))
        values = self.value_net(latent_vf) # (N, 1)

        distribution = self.get_distribution(observations)
        actions_taken = distribution.get_actions(deterministic=deterministic) # (N,)
        log_prob = distribution.log_prob(actions_taken)
        actions_taken = actions_taken.reshape((-1,) + self.action_space.shape)
        return actions_taken, values, log_prob

    def get_distribution(self, obs_and_mask_dict: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        Args:
            obs_and_mask_dict (th.Tensor): the observations and the mask
        
        Returns:
            Distribution: the current policy distribution
        """
        features_and_mask_dict = super().extract_features(obs_and_mask_dict) 
        mask = features_and_mask_dict["mask"] # (N,)
        latent_pi = self.mlp_extractor.forward_actor(features_and_mask_dict) # (N, pi_last_layer)
        logits = self.action_net(latent_pi) # (N, n_actions)
        logits -= 1e8 * (1 - mask) # Masking, (N, n_actions)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        return distribution
    
    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], state: Optional[Tuple[np.ndarray, ...]] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        observation = {key: th.as_tensor(observation[key], dtype=th.float32).unsqueeze(0) for key in observation}
        print("Shape:", {key: observation[key].shape for key in observation})
        # self.set_training_mode(False)
        B, N, n_actions = observation["mask"].shape
        observation = {key: observation[key].transpose(0, 1).reshape(-1, *observation[key].shape[2:]) for key in observation}
        print("Shape:", {key: observation[key].shape for key in observation})
        actions, state = self.get_distribution(observation).get_actions(deterministic=deterministic), state
        print(actions)
        actions = actions.reshape(B, N)
        return actions, state


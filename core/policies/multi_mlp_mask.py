
    
    
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


class MultiInputWithMaskExtractor(BaseFeaturesExtractor):
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
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        if 'mask' not in observation_space.spaces.keys():
            raise ValueError(f"Key {key} not found in observation space {observation_space} for MultiInputWithMaskExtractor. Please add a mask with corresponding key 'mask' to the observation space.")
        super().__init__(observation_space, features_dim=1)
        encoders = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key != "mask":
                # Get the shape of the observation
                shape = subspace.shape
                # We only deals with 1D observation
                if len(shape) != 1:
                    raise ValueError(f"Observation {key} is not 1D, got shape {shape} instead.")
                n_input = shape[0]

                # Create a linear layer to encode the input
                encoders[key] = nn.Sequential(
                    nn.Linear(n_input, 64), nn.ReLU(), 
                    nn.Linear(64, 64), nn.ReLU(),
                )

                # Update the total output size
                total_concat_size += 64

        self.encoders = nn.ModuleDict(encoders)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, dict_observations : Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """The forward pass of the features extractor.
        This apply a different encoders (a 2 layer MLP) to each observation (except the mask),
        then concatenate the output of each encoder and return the result.
        """
        list_batch_of_encoding = []
        for key, batch_of_partial_observations in dict_observations.items():  # (N, *dim_obs)
            if key != "mask":
                batch_of_encoding = self.encoders[key](batch_of_partial_observations)  # (N, 64)
                list_batch_of_encoding.append(batch_of_encoding)
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
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
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











class MultiInputWithMaskPolicy(ActorCriticPolicy):
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

        :param obs:
        :return: the action distribution.
        """
        features_and_mask_dict = super().extract_features(obs_and_mask_dict) 
        features = features_and_mask_dict["features"] # (N, 64 * n_observations)
        mask = features_and_mask_dict["mask"] # (N,)
        latent_pi = self.mlp_extractor.forward_actor(features_and_mask_dict) # (N, pi_last_layer)
        logits = self.action_net(latent_pi) # (N, n_actions)
        logits -= 1e8 * (1 - mask) # Masking, (N, n_actions)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        return distribution
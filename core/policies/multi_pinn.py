
    
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PermutationInvariantOperations(nn.Module):
    """ A NN that takes as input a tensor of variable size and outputs a tensor of fixed size.
    Formally, the input tensor has shape (B, N, dim_obs) and the output tensor has shape (B, 4*dim_obs).
    This is obtained by computing the max, min, sum and average of the N observations for each batch.
    """
    def __init__(self, dim=1):
        super(PermutationInvariantOperations, self).__init__()
        self.dim = dim
    
    def forward(self, x):  # (B, N, dim_obs)
        assert x.dim() == 3, "The input tensor must have shape (B, N, dim_obs)"
        max_vals, _ = th.max(x, dim=self.dim)
        min_vals, _ = th.min(x, dim=self.dim)
        sum_vals = th.sum(x, dim=self.dim)
        avg_vals = th.mean(x, dim=self.dim)
        output = th.cat([max_vals, min_vals, sum_vals, avg_vals], dim=1) # (B, 4*dim_obs)
        return output
    

class MultiInputPINNExtractor(BaseFeaturesExtractor):
    """The features extractor for the MultiInputPINN policy.
    The observations are a dict of tensors. Each of them may be (N>1) or not (N=1) a 'batched' feature,
    i.e. a feature that has to be seen as a set (permutation invariant) of a certain number N of observations, each of which is a vector of dimension dim_obs.
    obs = {
        'observation1': ,  # shape (B, N1, dim_obs1)
        'observation2': ,  # shape (B, N2, dim_obs2)
        'observation3': ,  # shape (B, dim_obs3)
        'observation4': ,  # shape (B, dim_obs4)
            }
    """
    def __init__(self, 
                observation_space: spaces.Dict, 
                n_neurons_encoder : int = 16, 
                n_neurons_batchable_encoder : int = 16, 
                n_neurons_batchable_hidden : int = 16):
        """Create a MultiInputPINNExtractor object.

        Args:
            observation_space (spaces.Dict): the observation space
            n_neurons_encoder (int, optional): the number of neurons for the encoder of non-batched observations. Defaults to 16.
            n_neurons_batchable_encoder (int, optional): the number of neurons for the output layer of the encoder of batched observations. Defaults to 16.
            n_neurons_batchable_hidden (int, optional): the number of neurons for the hidden layer of the encoder of batched observations. Defaults to 16.
        """
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        observation_encoders = {}
        batchable_observation_encoders = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
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
                raise ValueError(f"Observation {key} has shape {shape} which is not supported by MultiInputPINNExtractor. Please use a shape of (n,) or (m, n).")

        self.encoders = nn.ModuleDict(observation_encoders)
        self.batchable_encoders = nn.ModuleDict(batchable_observation_encoders)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, dict_observations : Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        list_batch_of_encoding = []
        for key, batch_of_partial_observations in dict_observations.items(): 
            if len(batch_of_partial_observations.shape) == 2:
                batch_of_encoding = self.encoders[key](batch_of_partial_observations)
                list_batch_of_encoding.append(batch_of_encoding)
            elif len(batch_of_partial_observations.shape) >= 3:
                batch_of_encoding = self.batchable_encoders[key](batch_of_partial_observations)
                list_batch_of_encoding.append(batch_of_encoding)
            else:
                raise ValueError(f"Observation {key} has shape {batch_of_partial_observations.shape} which is not supported by MultiInputPINNExtractor. Please use a shape of (n,) or (m, n).")
        batch_of_encoding_concat = th.cat(list_batch_of_encoding, dim=1)
        return batch_of_encoding_concat
    


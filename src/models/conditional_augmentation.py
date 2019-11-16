# The Conditonal Augmentation Component
# The component was not explained in the paper,
# thus we have use the implementation from PyTorch tutorial that can be found here:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningAugmentation(nn.Module):
    """ Conditioning Augmentation Block.
    Denoted with F^CA in the block diagram from the paper. 

    Parameters:
        input_dim (int): The input dim to the conditioning augmentation
            Should be the same as the output dimension of the text_encoder
            Thus, default: 256
        condition_dim (int): the dimension of the conditioning vector, default: 100

    TODO --->
    1. do not return also mu and std? they will be used in the future?"""

    def __init__(self, input_dim=256, condition_dim=100):
        super(ConditioningAugmentation, self).__init__()

        # Load pre-trained model tokenizer (vocabulary)
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        # The fully connected layer,
        # multiplied by 4 so after the GLU operation we can split half:
        # into mu and std in size of condition_dim
        self.fc = nn.Linear(input_dim, 4 * condition_dim)

    def forward(self, x):
        """ Performs conditioning augmentation using the reparametrization trick.
        Parameters:
            x (torch.Tensor): vector of shape input_dim, can be used as the output of the text encoder

        Returns:
            condition_vector (torch.Tensor of shape condition_dim):
                the "sampled" version of the text encoding
            mu (torch.Tensor of shape condition_dim): mean of the text encoding
            logvar (torch.Tensor of shape condition_dim): variance of the text encoding
        """
        # Compute the mean and stand deviation:
        pre_conditioning = F.glu(self.fc(x))
        mu = pre_conditioning[:, :self.condition_dim]
        log_var = pre_conditioning[:, self.condition_dim:]
        std = torch.exp(log_var/2)  # std must be non-zero

        # reparameterization trick:
        # multiply the std by a normal distributed noise and add the mean
        # in order to sample the conditionining without disabling the backpropagation
        # (cannot propagate through a random node)
        epsilon = torch.randn_like(std)
        condition_vector = mu + epsilon * std

        # Return all of the results
        return condition_vector, mu, log_var

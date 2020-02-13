""" influenced by:
    https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/\
    Image%20Recognition/utils/BBBlayers.py"""

import copy
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


class BayesianLayer(nn.Module):
    """ Bayesian wrapper for Fully connected and Convolution Layers.
    Args:
        layer (nn.Module): Pytorch Linear or Convolutional layer
        **kwargs :
            q_logvar_init (float): initial value for variance of q
            p_logvar_init (float): initial value for variance of p

    """

    def __init__(self, layer, alpha_shape=(1, 1), use_bias=False):
        super().__init__()
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.layer = deepcopy(layer)
        self.weight = Parameter(torch.zeros_like(self.layer.weight))
        if use_bias:
            self.bias = Parameter(torch.ones_like(self.layer.bias))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(sum(self.weight.size()[1:]))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):
        """
        estimating the weights distribution and update the \
        weights before calling the layer actual forward pass.

        Args:
            x (Tensor): input
        """

        raise NotImplementedError

    def regularization(self):
        kl = self.weight.nelement() / self.log_alpha.nelement() * calculate_kl(self.log_alpha)
        return kl

    def get_kwargs(self):
        kwargs = self.layer.__dict__
        wanted = ['stride', 'dilation', 'padding', 'groups']
        kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        return kwargs


class LinearBayesianLayer(BayesianLayer):
    def forward(self, x):
        """
        estimating the weights distribution and update the \
        weights before calling the layer actual forward pass.

        Args:
            x (Tensor): input

        Returns:
            x (Tensor): output
        """
        mean = F.linear(x, self.weight)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        epsilon = std.data.new(std.size()).normal_()
        # Local reparameterization trick
        out = mean + std * epsilon

        return out


class Conv1dBayesianLayer(BayesianLayer):
    def forward(self, x):
        """
        estimating the weights distribution and update the \
        weights before calling the layer actual forward pass.

        Args:
            x (Tensor): input

        Returns:
            x (Tensor): output
        """
        sig_weight = torch.exp(self.bias)
        weight = self.weight + sig_weight * self.eps_weight.normal_()
        x = F.conv1d(x, weight, bias=None)
        return x


class Conv2dBayesianLayer(BayesianLayer):
    def forward(self, x):
        """
        estimating the weights distribution and update the \
        weights before calling the layer actual forward pass.

        Args:
            x (Tensor): input

        Returns:
            x (Tensor): output
        """
        kwargs = self.get_kwargs()

        mean = F.conv2d(x, self.weight, self.bias, **kwargs)

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + F.conv2d(x * x, sigma, bias=None, **kwargs))
        epsilon = std.data.new(std.size()).normal_()

        out = mean + std * epsilon

        return out


def patch_module(module: torch.nn.Module, inplace: bool = True, types='all') -> torch.nn.Module:
    """Replace last layer in a model with Bayesian layer.

    Args:
        types:
        module (torch.nn.Module):
            The module in which you would like to replace dropout layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Returns:
        module (torch.nn.Module):
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)

    _patch_bayesian_layers(module, types=types)

    return module


def _patch_bayesian_layers(module: torch.nn.Module, types='all') -> None:
    """
    Recursively iterate over the children of a module and find the last
    layer to be wrapped by BayesianLayer.
    """
    assert types in ['all', 'linear', 'conv']
    layer_met = False
    for name, child in module.named_children():
        new_module = None
        if isinstance(child, nn.Linear) and types in ['all', 'linear']:
            layer_met = True
            new_module = LinearBayesianLayer(child)
        elif isinstance(child, nn.Conv2d) and types in ['all', 'conv']:
            layer_met = True
            new_module = Conv2dBayesianLayer(child)
        elif isinstance(child, nn.Conv1d) and types in ['all', 'conv']:
            layer_met = True
            new_module = Conv1dBayesianLayer(child)

        if isinstance(child, nn.Dropout):
            new_module = torch.nn.Dropout(p=0)

        if isinstance(child, nn.ReLU) and layer_met:
            new_module = nn.Softplus()
        if new_module is not None:
            module.add_module(name, new_module)
        _patch_bayesian_layers(child)


class BayesianModel(nn.Module):
    """ A class to define the regularization part for ELBO loss.

    Args:
        model (nn.Module): torch model
        q_logvar_init (float): initial value for variance of q
        p_logvar_init (float): initial value for variance of p

    """

    def __init__(self, model: nn.Module, types='all'):
        super(BayesianModel, self).__init__()
        self.model = patch_module(model, types=types)

    def forward(self, x):
        return self.model(x)

    def _iterate_layers(self, module):
        kl = 0
        for _, child in module.named_children():
            if isinstance(child, BayesianLayer):
                kl += child.regularization()
            kl += self._iterate_layers(child)
        return kl

    def regularization(self):
        kl = self._iterate_layers(self.model)
        return kl

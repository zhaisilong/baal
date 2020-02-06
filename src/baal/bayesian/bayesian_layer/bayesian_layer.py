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


class BayesianLayer(nn.Module):
    """ Bayesian wrapper for Fully connected and Convolution Layers.
    Args:
        layer (nn.Module): Pytorch Linear or Convolutional layer
        **kwargs :
            q_logvar_init (float): initial value for variance of q
            p_logvar_init (float): initial value for variance of p

    """

    def __init__(self, layer, q_logvar_init=0.5, p_logvar_init=0.5):
        super(BayesianLayer, self).__init__()

        self.q_logvar_init = torch.as_tensor(q_logvar_init)
        self.p_logvar_init = torch.as_tensor(p_logvar_init)

        self.layer = deepcopy(layer)
        self.weight_mu = Parameter(torch.zeros_like(self.layer.weight))
        self.weight_sigma = Parameter(torch.ones_like(self.layer.weight))
        self.register_buffer('eps_weight', torch.ones_like(self.layer.weight))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(1. / self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_sigma.data.fill_(self.p_logvar_init)
        self.eps_weight.data.zero_()

    def forward(self, x):
        """
        estimating the weights distribution and update the \
        weights before calling the layer actual forward pass.

        Args:
            x (Tensor): input
        """

        raise NotImplementedError

    def regularization(self):
        sig_weight = torch.exp(self.weight_sigma)
        kl_ = math.log(self.q_logvar_init) - self.weight_sigma + \
              (sig_weight ** 2 + self.weight_mu ** 2) / (2 * self.q_logvar_init ** 2) - 0.5
        kl = kl_.sum()
        return kl


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
        sig_weight = torch.exp(self.weight_sigma)
        weight = self.weight_mu + sig_weight * self.eps_weight.normal_()
        x = F.linear(x, weight, bias=None)
        return x


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
        sig_weight = torch.exp(self.weight_sigma)
        weight = self.weight_mu + sig_weight * self.eps_weight.normal_()
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
        sig_weight = torch.exp(self.weight_sigma)
        weight = self.weight_mu + sig_weight * self.eps_weight.normal_()
        kwargs = self.layer.__dict__
        wanted = ['stride', 'dilation', 'padding', 'groups']
        kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        x = F.conv2d(x, weight, bias=None, **kwargs)
        return x


def patch_module(module: torch.nn.Module, q_logvar_init=1.0, p_logvar_init=1.0,
                 inplace: bool = True, types='all') -> torch.nn.Module:
    """Replace last layer in a model with Bayesian layer.

    Args:
        types (str): One of 'all', 'linear', 'conv', the type of layers to modifies.
        module (torch.nn.Module):
            The module in which you would like to replace dropout layers.
        q_logvar_init (float): initial value for variance of q
        p_logvar_init (float): initial value for variance of p
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Returns:
        module (torch.nn.Module):
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)

    _patch_bayesian_layers(module, q_logvar_init=q_logvar_init, p_logvar_init=p_logvar_init,
                           types=types)

    return module


def _patch_bayesian_layers(module: torch.nn.Module, q_logvar_init=1.0, p_logvar_init=1.0,
                           types='all') -> None:
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
            new_module = LinearBayesianLayer(child,
                                             q_logvar_init=q_logvar_init,
                                             p_logvar_init=p_logvar_init)
        elif isinstance(child, nn.Conv2d) and types in ['all', 'conv']:
            layer_met = True
            new_module = Conv2dBayesianLayer(child,
                                             q_logvar_init=q_logvar_init,
                                             p_logvar_init=p_logvar_init)
        elif isinstance(child, nn.Conv1d) and types in ['all', 'conv']:
            layer_met = True
            new_module = Conv1dBayesianLayer(child,
                                             q_logvar_init=q_logvar_init,
                                             p_logvar_init=p_logvar_init)

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

    def __init__(self, model: nn.Module, q_logvar_init=1.0, p_logvar_init=1.0, types='all'):
        super(BayesianModel, self).__init__()
        self.model = patch_module(model, q_logvar_init=q_logvar_init, p_logvar_init=p_logvar_init,
                                  types=types)

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

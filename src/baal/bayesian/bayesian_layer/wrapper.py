from copy import deepcopy
from typing import Optional, Callable

import structlog
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset

from baal.modelwrapper import ModelWrapper

log = structlog.get_logger("ModelWrapper")


class BayesianWrapper(ModelWrapper):
    """
    A class to do bayesian inference using a model with bayesian layers.
    This class assures that a model with Bayesian Layers accompanies is
    followed by a Softplus() instead of ReLu().
    It also makes sure that we have few samples of the training output,
    to be able to estimate the uncertainty of loss through VI.

    Args:
        model : (torch.nn.Module) pytorch model
        criterion : (torch.nn.Module) pytorch loss function
        beta : scale for regularization effect
        iterations: The number of times the model would iterate over
            the batch to create uncertainty.

    Returns:
        Tensor , computed bayesian loss
    """

    def __init__(self, model, criterion, beta=1e-4):
        super(BayesianWrapper, self).__init__(model, criterion)

        self.beta = beta

    def train_on_batch(self, data, target, optimizer, cuda=False):
        beta = self.beta / data.size(0)

        if cuda:
            data, target = data.cuda(), target.cuda()
            self.model.cuda()
        optimizer.zero_grad()
        output = self.model(data)
        regularizer = self.model.regularization()

        loss = self.criterion(output, target) + beta * regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 50)
        # print("total_norm_after: ", total_norm)
        optimizer.step()
        self._update_metrics(output, target, loss, filter='train')
        optimizer.zero_grad()
        # print("are we ok with the list of loss:", self.metrics['train_loss'].loss)
        # print("are we ok with the avg loss:", self.metrics['train_loss'].value)
        return loss

    def test_on_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        cuda: bool = False,
        average_predictions=1
    ):
        """
        Test the current model on a batch.

        Args:
            data (Tensor): the model input
            target (Tensor): the ground truth
            cuda (bool): use cuda or not

        Returns:
            Tensor, the loss computed from the criterion.
        """
        beta = self.beta / data.size(0)
        with torch.no_grad():
            if cuda:
                data, target = data.cuda(), target.cuda()
            if average_predictions == 1:
                preds = self.model(data)
                loss = self.criterion(preds, target)
            elif average_predictions > 1:
                raise NotImplementedError
            self._update_metrics(preds.cpu(), target.cpu(), loss, 'test')
            return loss

    def reset_last_module(self):
        def reset(m):
            _, child = list(zip(*m.named_childern()))[-1]
            if len(list(child.named_children())) == 0:
                child.reset_parameters()
            reset(child)

        self.model.apply(reset)

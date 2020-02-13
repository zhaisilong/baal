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
        if cuda:
            data, target = data.cuda(), target.cuda()
            self.model.cuda()
        optimizer.zero_grad()
        output = self.model(data)
        regularizer = self.model.regularization()

        loss = self.criterion(output, target) + self.beta * regularizer
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), 50)
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

    def train_and_test_on_datasets(self, train_dataset: Dataset, test_dataset: Dataset,
                                   optimizer: Optimizer, batch_size: int, epoch: int,
                                   use_cuda: bool,
                                   workers: int = 4,
                                   collate_fn: Optional[Callable] = None,
                                   return_best_weights=False,
                                   patience=None,
                                   min_epoch_for_es=0,
                                   test_every=1,
                                   lr_schedule=None):
        """
        Train and test the model on both Dataset `train_dataset`, `test_dataset`.

        Args:
            train_dataset (Dataset): Dataset to train on.
            test_dataset (Dataset): Dataset to evaluate on.
            optimizer (Optimizer): Optimizer to use during training.
            batch_size (int): Batch size used.
            epoch (int): number of epoch to train on.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            return_best_weights (bool): If True, will keep the best weights and return them.
            patience (Optional[int]): If provided, will use early stopping to stop after
                                        `patience` epoch without improvement.
            min_epoch_for_es (int): Epoch at which the early stopping starts.

        Returns:
            History and best weights if required.
        """
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(epoch):
            _ = self.train_on_dataset(train_dataset, optimizer, batch_size, test_every,
                                      use_cuda, workers, collate_fn)
            te_loss = self.test_on_dataset(test_dataset, batch_size, use_cuda, workers, collate_fn)
            if lr_schedule:
                lr_schedule.step()
            hist.append({k: v.value for k, v in self.metrics.items()})
            if te_loss < best_loss:
                best_epoch = e
                best_loss = te_loss
                if return_best_weights:
                    best_weight = deepcopy(self.state_dict())

            if patience is not None and (e - best_epoch) > patience and (e > min_epoch_for_es):
                # Early stopping
                break

        if return_best_weights:
            return hist, best_weight
        else:
            return hist

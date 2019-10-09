import torch
import structlog
from torch.autograd import Variable

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
    def __init__(self, model, criterion, beta=1):
        super(BayesianWrapper, self).__init__(model, criterion)

        self.beta = beta

    def train_on_batch(self, data, target, optimizer, cuda=False):

        data, target = Variable(data), Variable(target)
        beta = self.beta / data.size(0)

        if cuda:
            data, target = data.cuda(), target.cuda()
            self.model.cuda()
        optimizer.zero_grad()
        outputs = self.model(data)
        regularizer = Variable(self.model.regularization())

        if cuda:
            regularizer = regularizer.cuda()

        loss = self.criterion(outputs, target) + \
               beta * regularizer

        print("loss before:", loss)
        loss.backward()

        # gradient clip to prevent gradient overflow
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
        optimizer.step()
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

                # the regularization loss is used only for training
                return self.criterion(self.model(data), target)
            else:
                raise Exception("BayesianWrapper doesn't support this functionality")

    def predict_on_batch(self, data, cuda=False, iterations=1, average_predictions=1):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): the model input
            iterations (int): number of prediction to perform.
            cuda (bool): use cuda or not

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}
        """
        with torch.no_grad():
            if cuda:
                data = data.cuda()

            if average_predictions == 1:
                outputs = []
                for _ in range(iterations):
                    out = self.model(data)
                    outputs.append(out)
                return torch.stack(outputs, dim=-1)
            else:
                raise Exception("BayesianWrapper doesn't support this functionality")

    def reset_last_module(self):
        def reset(m):
            _, child = list(zip(*m.named_childern()))[-1]
            if len(list(child.named_children())) == 0:
                child.reset_parameters()
            reset(child)
        self.model.apply(reset)



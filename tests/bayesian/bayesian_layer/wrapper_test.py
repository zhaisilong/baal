import pytest
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from baal.bayesian.bayesian_layer import BayesianModel
from baal.bayesian.bayesian_layer.wrapper import BayesianWrapper


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.l1 = nn.Linear(10, 8)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x


class DummyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([10]) * item / 255.).float(), torch.FloatTensor([item % 2])


class BayesianWrapperTest(unittest.TestCase):
    def setUp(self):
        self.model = BayesianModel(DummyModel())

        self.criterion = nn.MSELoss()

        self.wrapper = BayesianWrapper(self.model, self.criterion)
        self.optim = torch.optim.SGD(self.wrapper.get_params(), 0.01)
        self.dataset = DummyDataset()

    def test_train_on_batch(self):
        with torch.autograd.set_detect_anomaly(True):
            self.wrapper.train()
            old_param = list(map(lambda x: x.clone(), self.model.parameters()))
            input, target = torch.randn(1, 10), torch.randn(1, 1)
            self.wrapper.train_on_batch(input, target, self.optim)
            new_param = list(map(lambda x: x.clone(), self.model.parameters()))
            assert any([not torch.allclose(i, j) for i, j in zip(old_param, new_param)])

    def test_predict_on_batch(self):
        self.wrapper.eval()
        input = torch.randn(2, 10)

        # iteration == 10
        pred1 = self.wrapper.predict_on_batch(input, iterations=10, cuda=False)
        pred2 = self.wrapper.predict_on_batch(input, iterations=10, cuda=False)
        assert not torch.allclose(pred1, pred2)

        # check the predictions are different for different iterations of same batch
        assert any([not torch.allclose(pred1[..., i], pred1[..., j]) for i in range(10) for j in range(10)])

    def test_reset_last_module(self):
        last_module = self.model.model.l2

        last_module_params = list(map(lambda x: x.clone(), last_module.parameters()))

        input, target = torch.randn(1, 10), torch.randn(1, 1)
        self.wrapper.train_on_batch(input, target, self.optim)
        after_train_params = list(map(lambda x: x.clone(), last_module.parameters()))

        self.wrapper.reset_last_module()
        new_last_module_params = list(map(lambda x: x.clone(), last_module.parameters()))


        assert any([not torch.allclose(i, j) for i, j in zip(last_module_params, after_train_params)])
        assert any([not torch.allclose(i, j) for i, j in zip(after_train_params, new_last_module_params)])

        # parameters of bayesian_layer would reintialize randomly but not to the initial values
        assert any ([not torch.allclose(i, j) for i, j in zip(last_module_params, new_last_module_params)])

if __name__ == '__main__':
    pytest.main()

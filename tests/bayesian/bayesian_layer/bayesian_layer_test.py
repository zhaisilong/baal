import unittest
import pytest
import torch.nn as nn

from baal.bayesian.bayesian_layer import BayesianModel, BayesianLayer


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.l1 = nn.Linear(3, 10)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(10, 20)

    def forward(self, x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x


class BayesianWrapperTest(unittest.TestCase):
    def setUp(self):
        model = DummyModel()
        self.bayes_model = BayesianModel(model)

    def test_layer_is_changed(self):
        assert isinstance(self.bayes_model.model.l1, BayesianLayer)
        assert isinstance(self.bayes_model.model.r1, nn.Softplus)
        assert isinstance(self.bayes_model.model.l2, BayesianLayer)
        assert hasattr(self.bayes_model.model.l1, 'regularization')
        assert hasattr(self.bayes_model.model.l2, 'regularization')


if __name__ == '__main__':
    pytest.main()

import torch
from torch import nn
import numpy as np

from utils.pareto import batched_pareto_dominates


# Based on implementation from: https://github.com/zpschang/DPMORL
# Credits for the original code go to the author: Xin-Qiang Cai


class UtilityFunction(nn.Module):
    def __init__(
            self,
            min_val,
            max_val,
            frozen=True,
            normalise=True,
            max_weight=0.1,
            size_factor=1
    ):
        super().__init__()
        # Initialize the variables
        self.reward_shape = len(min_val)
        self.min_val = torch.tensor(min_val)
        self.max_val = torch.tensor(max_val)
        self.normalise = normalise

        # Initialize the utility function
        self.mlp1 = nn.Linear(self.reward_shape, 24 * size_factor)
        self.mlp2 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp3 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp4 = nn.Linear(72 * size_factor, 1)
        self.max_weight = max_weight

        # Initialize positive weight
        self.make_monotone_init()
        self.make_monotone()
        if frozen:
            self.make_frozen()

        # Compute min and max utility values
        self.min_u, self.max_u = self.compute_utility(torch.stack([min_val, max_val]))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        utilities = self.compute_utility(x)
        if self.normalise:
            utilities = (utilities - self.min_u) / (self.max_u - self.min_u)
        return utilities

    def compute_utility(self, input_x):
        x = self.mlp1(input_x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp2(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp3(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp4(x)
        return x[:, 0]

    def make_monotone_init(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = layer.weight.data.abs()

    def make_monotone(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = torch.maximum(layer.weight.data, torch.tensor(0.0))
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4]:
            layer.weight.data = torch.minimum(layer.weight.data, torch.tensor(self.max_weight))

    def make_frozen(self):
        for param in self.parameters():
            param.requires_grad = False


def test_monotonicity(num_tests=20, num_samples_per_test=10):
    """Test the monotonicity property of the generated utility functions."""
    min_val = torch.tensor([0, 0], dtype=torch.float32)
    max_val = torch.tensor([10, 10], dtype=torch.float32)

    for test in range(num_tests):
        uf = UtilityFunction(min_val, max_val, normalise=True, max_weight=0.1, size_factor=1)
        samples = torch.round(torch.rand(num_samples_per_test, 2) * 10, decimals=4)
        utilities = uf(samples).numpy()
        np_samples = samples.numpy()

        for sample, utility in zip(np_samples, utilities):
            mask = batched_pareto_dominates(sample, np_samples)
            masked_utilities = utilities * mask
            assert np.all(utility >= masked_utilities)

    print("All tests passed!")


if __name__ == '__main__':
    test_monotonicity(num_tests=20, num_samples_per_test=100)

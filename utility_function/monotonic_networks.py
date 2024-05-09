import torch
from torch import nn
import torch.nn.functional as F


class MNN(nn.Module):
    """
    A regular monotonic neural network utility function.
    """

    def __init__(
            self,
            min_vec,
            max_vec,
            scale_in=True,
            scale_out=True,
            frozen=True,
            max_weight=0.1,
    ):
        super().__init__()
        # Initialize the variables
        self.reward_shape = len(min_vec)
        self.min_vec = torch.tensor(min_vec, dtype=torch.float32)
        self.max_vec = torch.tensor(max_vec, dtype=torch.float32)
        self.scale_in = scale_in
        self.scale_out = scale_out

        # Initialize the utility function
        hidden_dims = (64, 128, 64)
        activation_fn = nn.Sigmoid
        self.layers = nn.Sequential(
            nn.Linear(self.reward_shape, hidden_dims[0]),
            activation_fn(),
        )
        for d_in, d_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([
                nn.Linear(d_in, d_out),
                activation_fn(),
            ])
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.max_weight = max_weight

        # Initialize positive weight
        self.make_monotone_init()
        if frozen:
            self.make_frozen()

        # Compute min and max utility values
        if self.scale_in:
            min_in_vec = torch.zeros(self.reward_shape, dtype=torch.float32)
            max_in_vec = torch.ones(self.reward_shape, dtype=torch.float32)
        else:
            min_in_vec = self.min_vec
            max_in_vec = self.max_vec

        self.min_u, self.max_u = self.compute_utility(torch.stack([min_in_vec, max_in_vec]))
        print(self.min_u, self.max_u)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        if self.scale_in:
            x = (x - self.min_vec) / (self.max_vec - self.min_vec)

        utilities = self.compute_utility(x)

        if self.scale_out:
            utilities = (utilities - self.min_u) / (self.max_u - self.min_u)
        return utilities.squeeze()

    def compute_utility(self, input_x):
        return self.layers(input_x)

    def make_monotone_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.abs()
                layer.weight.data = torch.minimum(layer.weight.data, torch.tensor(self.max_weight))

    def make_frozen(self):
        for param in self.parameters():
            param.requires_grad = False


class MNNCai(nn.Module):
    """
    Based on implementation from: https://github.com/zpschang/DPMORL. Credits for the original code go to the
    author Xin-Qiang Cai.

    Note:
        This utility function generates a poor spread in my opinion.
    """

    def __init__(
            self,
            min_vec,
            max_vec,
            scale_in=True,
            scale_out=True,
            frozen=True,
            max_weight=0.1,
            size_factor=1
    ):
        super().__init__()
        # Initialize the variables
        self.reward_shape = len(min_vec)
        self.min_vec = torch.tensor(min_vec, dtype=torch.float32)
        self.max_vec = torch.tensor(max_vec, dtype=torch.float32)
        self.scale_in = scale_in
        self.scale_out = scale_out

        # Initialize the utility function
        self.mlp1 = nn.Linear(self.reward_shape, 24 * size_factor)
        self.mlp2 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp3 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.mlp4 = nn.Linear(72 * size_factor, 1)
        self.max_weight = max_weight

        # Initialize positive weight
        self.make_monotone_init()
        # self.make_monotone()
        if frozen:
            self.make_frozen()

        # Compute min and max utility values
        if self.scale_in:
            min_in_vec = torch.zeros(self.reward_shape, dtype=torch.float32)
            max_in_vec = torch.ones(self.reward_shape, dtype=torch.float32)
        else:
            min_in_vec = self.min_vec
            max_in_vec = self.max_vec

        self.min_u, self.max_u = self.compute_utility(torch.stack([min_in_vec, max_in_vec]))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        if self.scale_in:
            x = (x - self.min_vec) / (self.max_vec - self.min_vec)

        utilities = self.compute_utility(x)

        if self.scale_out:
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


def base(x):
    return F.selu(x)


def base_hat(x):
    return -base(-x)


def base_tilde(x):
    x_neg = x < 0
    out = (base(x + 1) - base(torch.ones_like(x))) * x_neg + (base_hat(x - 1) + base(torch.ones_like(x))) * ~x_neg
    return out


def combined_activation_fn(x, s):
    x0 = x[:, :s[0]]
    x1 = x[:, s[0]:s[0] + s[1]]
    x2 = x[:, s[0] + s[1]:]

    o0 = base(x0)
    o1 = base_hat(x1)
    o2 = base_tilde(x2)
    x_out = torch.cat([o0, o1, o2], -1)
    return x_out


class ConstrainedMNN(nn.Module):
    """
    Based on architecture from: https://arxiv.org/abs/2205.11775
    """

    def __init__(
            self,
            min_vec,
            max_vec,
            scale_in=True,
            scale_out=True,
            frozen=True,
            max_weight=0.1,
    ):
        super().__init__()
        # Initialize the variables
        self.reward_shape = len(min_vec)
        self.min_vec = torch.tensor(min_vec, dtype=torch.float32)
        self.max_vec = torch.tensor(max_vec, dtype=torch.float32)
        self.scale_in = scale_in
        self.scale_out = scale_out

        # Initialize the utility function
        hidden_dim = 64
        self.linear1 = nn.Linear(self.reward_shape, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)
        split = [0, 0, 64]
        self.s1 = torch.tensor(split)
        self.s2 = torch.tensor(split)
        self.s3 = torch.tensor(split)
        self.s4 = torch.tensor(split)
        self.max_weight = max_weight

        # Initialize positive weight
        self.make_monotone()
        if frozen:
            self.make_frozen()

        # Compute min and max utility values
        if self.scale_in:
            min_in_vec = torch.zeros(self.reward_shape, dtype=torch.float32)
            max_in_vec = torch.ones(self.reward_shape, dtype=torch.float32)
        else:
            min_in_vec = self.min_vec
            max_in_vec = self.max_vec

        self.min_u, self.max_u = self.compute_utility(torch.stack([min_in_vec, max_in_vec]))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        if self.scale_in:
            x = (x - self.min_vec) / (self.max_vec - self.min_vec)

        utilities = self.compute_utility(x)

        if self.scale_out:
            utilities = (utilities - self.min_u) / (self.max_u - self.min_u)
        return utilities.squeeze(-1)

    def compute_utility(self, input_x):
        x = self.linear1(input_x)
        x = combined_activation_fn(x, self.s1)
        x = self.linear2(x)
        x = combined_activation_fn(x, self.s2)
        x = self.linear3(x)
        x = combined_activation_fn(x, self.s3)
        x = self.linear4(x)
        x = combined_activation_fn(x, self.s4)
        x = self.linear5(x)
        return x

    def make_monotone(self):
        for layer in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
            layer.weight.data = layer.weight.data.abs()
            layer.weight.data = torch.minimum(layer.weight.data, torch.tensor(self.max_weight))

    def make_frozen(self):
        for param in self.parameters():
            param.requires_grad = False

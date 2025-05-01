import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, layers, activation="ReLU"):
        super(FullyConnectedNet, self).__init__()

        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sine": torch.sin  # Use with caution; not nn.Module
        }

        act_fn = activations.get(activation, nn.ReLU())

        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f"Linear_{i}", nn.Linear(layers[i], layers[i + 1]))
            if activation == "Sine":
                self.net.add_module(f"Sine_{i}", nn.Tanh())  # placeholder for sine
            else:
                self.net.add_module(f"{activation}_{i}", act_fn)
        self.net.add_module("Output", nn.Linear(layers[-2], layers[-1]))

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if callable(self.net[-2]):  # If Sine is implemented later
            return torch.sin(self.net[:-1](x))
        return self.net(x)

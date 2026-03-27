import torch as th
import torch.nn as nn
from gymnasium.spaces import Box

class GridNetwork(nn.Module):
    def __init__(self, observation_space: Box):
        super(GridNetwork, self).__init__()

        # extract subspaces
        grid_shape = observation_space.shape # grid is global observation
        n_input_channels = grid_shape[2]

        # grid network
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, observations):
        return self.cnn(observations)

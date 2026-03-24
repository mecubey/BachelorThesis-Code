import torch as th
import torch.nn as nn
from gymnasium.spaces import Box

class CriticNetwork(nn.Module):
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super(CriticNetwork, self).__init__()

        # extract subspaces
        grid_shape = observation_space.shape # grid is global observation
        n_input_channels = grid_shape[2]

        # grid network
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute shape by doing a single forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)  # from NHWC to NCHW 
            cnn_out_dim = self.cnn(sample).shape[1]

        # final layer (get value of state)
        self.final = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
        )

    def forward(self, observations):
        # convert grid to NCHW for CNN
        observations = observations.permute(0, 3, 1, 2)

        grid_features = self.cnn(observations)

        return self.final(grid_features)

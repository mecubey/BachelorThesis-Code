import torch as th
import torch.nn as nn
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # extract subspaces
        grid_shape = observation_space["grid"].shape
        one_hot_shape = observation_space["one_hot"].shape

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
            sample = th.as_tensor(observation_space["grid"].sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)  # from NHWC to NCHW 
            cnn_out_dim = self.cnn(sample).shape[1]

        # one hot network
        self.one_hot_net = nn.Sequential(
            nn.Linear(one_hot_shape[0], 32),
            nn.ReLU()
        )

        # final layer
        self.final = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations):
        grid = observations["grid"]
        one_hot = observations["one_hot"]

        # Convert grid to NCHW for CNN
        grid = grid.permute(0, 3, 1, 2)

        grid_features = self.cnn(grid)
        one_hot_features = self.one_hot_net(one_hot)

        combined = th.cat([grid_features, one_hot_features], dim=1)
        return self.final(combined)

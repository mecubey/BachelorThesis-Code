import torch as th
import torch.nn as nn
from gymnasium.spaces import Dict

class ActorNetwork(nn.Module):
    def __init__(self, observation_space: Dict, action_dim: int, features_dim: int = 256):
        super(ActorNetwork, self).__init__()

        # observation_space is a dictionary containing
        # {"grid": ..., "one_hot": ...}

        # extract subspaces
        grid_shape = observation_space["grid"].shape # grid is global observation
        one_hot_shape = observation_space["one_hot"].shape

        n_input_channels = grid_shape[2]

        # grid network
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(one_hot_shape[0], 4),
            nn.ReLU()
        )

        # final layer (get action logits)
        self.final = nn.Sequential(
            nn.Linear(cnn_out_dim + 4, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, action_dim),
        )

    def forward(self, observations):
        grid = observations["grid"]
        one_hot = observations["one_hot"]

        # convert grid to NCHW for CNN
        grid = grid.permute(0, 3, 1, 2)

        grid_features = self.cnn(grid)
        one_hot_features = self.one_hot_net(one_hot)

        combined = th.cat([grid_features, one_hot_features], dim=1)
        return self.final(combined)

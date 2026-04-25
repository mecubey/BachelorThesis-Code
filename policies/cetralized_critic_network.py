import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    def __init__(self, h, w, c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.conv(dummy).shape[1]

        self.value_head = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs):
        global_obs = global_obs.permute(0, 3, 1, 2)
        x = self.conv(global_obs)
        value = self.value_head(x)
        return value
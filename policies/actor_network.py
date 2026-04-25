import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, h, w, c, vec_dim, num_actions=5):
        super().__init__()

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, w, h)
            grid_out_dim = self.grid_encoder(dummy).shape[1]

        self.vec_encoder = nn.Sequential(
            nn.Linear(vec_dim, 32),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(grid_out_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, grid_obs, vec_obs, masks):
        grid_obs = grid_obs.permute(0, 3, 1, 2) # B, H, W, C -> B, C, H, W
        g = self.grid_encoder(grid_obs)
        v = self.vec_encoder(vec_obs)

        x = torch.cat([g, v], dim=-1)
        logits = self.policy_head(x)

        return logits.masked_fill(masks == 0, -1e9)
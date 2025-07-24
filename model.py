import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 9 * 9, 128), nn.ReLU(),
            nn.Linear(128, 81)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

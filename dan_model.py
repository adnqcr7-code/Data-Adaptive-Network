import torch
import torch.nn as nn

CODE_SIZE = 512  # the "recipe" length per image

class DANEncoder(nn.Module):
    """Image -> small code"""
    def __init__(self, code_size=CODE_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 16 -> 8
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 8 * 8, code_size)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class DANDecoder(nn.Module):
    """small code -> reconstructed image"""
    def __init__(self, code_size=CODE_SIZE):
        super().__init__()
        self.fc = nn.Linear(code_size, 256 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, code):
        x = self.fc(code)
        x = x.view(-1, 256, 8, 8)
        return self.net(x)


class DAN(nn.Module):
    def __init__(self, code_size=CODE_SIZE):
        super().__init__()
        self.encoder = DANEncoder(code_size)
        self.decoder = DANDecoder(code_size)

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code), code

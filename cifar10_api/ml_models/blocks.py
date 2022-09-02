import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResidualBlock(nn.Module):
    def __init__(self, blocks, shortcut=None):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        self.shortcut = shortcut

    def forward(self, x):
        return self.blocks(x) + (self.shortcut(x) if self.shortcut else x)

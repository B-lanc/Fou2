import torch.nn as nn
import torch.nn.functional as F

from .Blocks import ConvBlock
from . import FFT


class FourierUnit(nn.Module):
    def __init__(self, channels, width, height, freeze_parameters):
        super(FourierUnit, self).__init__()

        self.fft_w = FFT(width, freeze_parameters)
        self.fft_h = FFT(height, freeze_parameters)

        self.block = ConvBlock(channels, channels)

    def forward(self, x):
        return x


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralTransform, self).__init__()

        self.f

        self.out = nn.Conv2d(out_channels, out_channels, 1, 1, 0)


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFC, self).__init__()

        self.l_l = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.l_g = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.g_l = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.g_g = SpectralTransform(in_channels, out_channels)

        self.l_norm = nn.GroupNorm(8, out_channels)
        self.g_norm = nn.GroupNorm(8, out_channels)

    def forward(self, xl, xg):

        ll = self.l_l(xl)
        lg = self.l_g(xl)
        gl = self.g_l(xg)
        gg = self.g_g(xg)

        xl = ll + gl
        xg = lg + gg

        xl = F.relu(self.l_norm(xl))
        xg = F.relu(self.g_norm(xg))

        return xl, xg

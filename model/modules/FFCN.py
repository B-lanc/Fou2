import torch
import torch.nn as nn
import torch.nn.functional as F

from .Blocks import ConvBlock
from .FFT import FFT


class FourierUnit(nn.Module):
    def __init__(self, channels, fft_w, fft_h, chgn, dropout):
        super(FourierUnit, self).__init__()

        self.fft_w = fft_w
        self.fft_h = fft_h

        self.block = ConvBlock(2 * channels, 2 * channels, chgn, dropout, 1, 1, 0)

    def _forward(self, x):
        (_, ch, _, _) = x.shape
        yr, yi = self.rfft(x)
        y = torch.cat((yr, yi), dim=1)
        y = self.block(y)
        yr, yi = y[:, :ch, :, :], y[:, ch:, :, :]
        x = self.irfft(yr, yi)
        return x

    def _forward2(self, x):
        (_, ch, _, _) = x.shape
        y = torch.fft.rfftn(x, dim=[3, 2])
        yr, yi = y.real, y.imag
        y = torch.cat((yr, yi), dim=1)
        y = self.block(y)
        yr, yi = y[:, :ch, :, :], y[:, ch:, :, :]
        y = torch.complex(yr, yi)
        x = torch.fft.irfftn(y, dim=[3, 2])
        return x

    def forward(self, x):
        (_, _, h, _) = x.shape
        bot = x[:, :, :-1, :]
        top = x[:, :, 1:, :]

        bot = self._forward(bot)
        top = self._forward(top)

        x = torch.cat((bot[:, :, : h // 2, :], top[:, :, h // 2 - 1 :, :]), dim=2)
        return x

    def rfft(self, x):
        (_, _, _, w) = x.shape

        yr, yi = self.fft_w(x, None)
        yr, yi = yr[:, :, :, : w // 2 + 1], yi[:, :, :, : w // 2 + 1]
        yr, yi = yr.permute(0, 1, 3, 2), yi.permute(0, 1, 3, 2)

        yr, yi = self.fft_h(yr, yi)
        yr, yi = yr.permute(0, 1, 3, 2), yi.permute(0, 1, 3, 2)

        return yr, yi

    def irfft(self, xr, xi):
        yr, yi = xr.permute(0, 1, 3, 2), xi.permute(0, 1, 3, 2)
        yr, yi = self.fft_h(yr, yi, True)

        yr, yi = yr.permute(0, 1, 3, 2), yi.permute(0, 1, 3, 2)
        yr = torch.cat((yr, torch.flip(yr[:, :, :, 1:-1], dims=[3])), dim=3)
        yi = torch.cat((yi, -torch.flip(yi[:, :, :, 1:-1], dims=[3])), dim=3)
        yr, _ = self.fft_w(yr, yi, True)

        return yr


class SpectralTransform(nn.Module):
    def __init__(self, channels, chgn, dropout, fft_w, fft_h):
        super(SpectralTransform, self).__init__()

        self.conv_in = ConvBlock(channels, channels, chgn, dropout, 1, 1, 0)
        self.fu = FourierUnit(channels, fft_w, fft_h, chgn, dropout)
        self.conv_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv_in(x)
        y = self.fu(x)
        y = y + x
        y = self.conv_out(y)
        return y


class FFC(nn.Module):
    def __init__(self, channels, chgn, dropout, fft_w, fft_h):
        super(FFC, self).__init__()

        self.l_l = nn.Conv2d(channels, channels, 3, 1, 1)
        self.l_g = nn.Conv2d(channels, channels, 3, 1, 1)
        self.g_l = nn.Conv2d(channels, channels, 3, 1, 1)
        self.g_g = SpectralTransform(channels, chgn, dropout, fft_w, fft_h)

        self.l_norm = nn.GroupNorm(chgn, channels)
        self.g_norm = nn.GroupNorm(chgn, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xl, xg):

        ll = self.l_l(xl)
        lg = self.l_g(xl)
        gl = self.g_l(xg)
        gg = self.g_g(xg)

        xl = ll + gl
        xg = lg + gg

        xl = self.dropout(F.relu(self.l_norm(xl)))
        xg = self.dropout(F.relu(self.g_norm(xg)))

        return xl, xg

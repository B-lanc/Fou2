import torch
import torch.nn as nn
import numpy as np


class FFT(nn.Module):
    """
    FFT on the last axis
    """

    def __init__(self, window_size, freeze_parameters=True):
        super(FFT, self).__init__()

        kn = np.arange(window_size)[:, None] * np.arange(window_size)
        ohm = np.exp(-2 * np.pi * 1j / window_size)
        iohm = np.exp(2 * np.pi * 1j / window_size)

        w = np.power(ohm, kn)
        iw = np.power(iohm, kn) / window_size

        self.conv_real = nn.Conv1d(1, window_size, window_size, 1, 0)
        self.conv_imag = nn.Conv1d(1, window_size, window_size, 1, 0)
        self.iconv_real = nn.Conv1d(1, window_size, window_size, 1, 0)
        self.iconv_imag = nn.Conv1d(1, window_size, window_size, 1, 0)

        self.conv_real.weight.data = torch.Tensor(w.real[:, None, :])
        self.conv_imag.weight.data = torch.Tensor(w.imag[:, None, :])
        self.iconv_real.weight.data = torch.Tensor(iw.real[:, None, :])
        self.iconv_imag.weight.data = torch.Tensor(iw.imag[:, None, :])

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x_r, x_i, inverse=False):
        if x_i is None:
            x_i = torch.zeros_like(x_r)
        conv_real = self.iconv_real if inverse else self.conv_real
        conv_imag = self.iconv_imag if inverse else self.conv_imag

        shape = list(x_r.shape)
        _last = shape[-1]
        _batch = np.prod(shape[:-1])

        x_r = x_r.reshape(_batch, 1, _last)
        x_i = x_i.reshape(_batch, 1, _last)

        real = conv_real(x_r) - conv_imag(x_i)
        imag = conv_imag(x_r) + conv_real(x_i)
        # (_batch, chout, 1)

        real = real.reshape(shape)
        imag = imag.reshape(shape)
        return real, imag

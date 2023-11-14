import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class STFT(nn.Module):
    def __init__(
        self,
        nfft,
        hop_size,
        freeze_parameters=True,
        padding=True,
        pad_mode="reflect",  # reflect or constant
        scaled=False,
    ):
        super(STFT, self).__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.padding = padding
        self.pad_mode = pad_mode
        self.scaled = scaled

        if not hop_size:
            self.hop_size = self.nfft // 4

        hann = librosa.filters.get_window("hann", self.nfft, fftbins=True)

        ohm = np.exp(-2 * np.pi * 1j / self.nfft)
        w = np.matmul(
            np.arange(self.nfft)[:, None], np.arange(self.nfft // 2 + 1)[None, :]
        )
        self.w = np.power(ohm, w) * hann[:, None]  # (nfft, nfft//2+1)

        self.conv_real = nn.Conv1d(
            1, self.nfft // 2 + 1, self.nfft, self.hop_size, bias=False
        )
        self.conv_imag = nn.Conv1d(
            1, self.nfft // 2 + 1, self.nfft, self.hop_size, bias=False
        )
        self.conv_real.weight.data = torch.Tensor(self.w.real.T)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(self.w.imag.T)[:, None, :]

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        """
        x => (bs, length)
        if multichannel/stereo, just add it as another batch
        """
        assert x.dim() == 2
        x = x[:, None, :]

        if self.padding:
            x = F.pad(x, pad=(self.nfft // 2, self.nfft // 2), mode=self.pad_mode)
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (bs, nfft//2+1, nframes)
        if self.scaled:
            real = real * 2 / self.nfft
            imag = imag * 2 / self.nfft
        return real, imag


class ISTFT(nn.Module):
    def __init__(
        self,
        nfft,
        hop_size,
        freeze_parameter=True,
        scaled=False,
    ):
        super(ISTFT, self).__init__()
        self.nfft = nfft
        self.hop_size = hop_size

        if not hop_size:
            self.hop_size = self.nfft // 4

        hann = librosa.filters.get_window("hann", self.nfft, fftbins=True)
        ohm = np.exp(2 * np.pi * 1j / self.nfft)
        w = np.matmul(
            np.arange(self.nfft)[:, None],
            np.arange(self.nfft)[None, :],
        )
        w = np.power(ohm, w) * hann[None, :]  # (nfft, nfft)
        self.w = w / 2 if scaled else w / self.nfft

        self.conv_real = nn.Conv1d(self.nfft, self.nfft, 1, 1, bias=False)
        self.conv_imag = nn.Conv1d(self.nfft, self.nfft, 1, 1, bias=False)
        self.conv_real.weight.data = torch.Tensor(self.w.real.T)[:, :, None]
        self.conv_imag.weight.data = torch.Tensor(self.w.imag.T)[:, :, None]
        # (nfft, nfft, 1)

        # overlap add window
        ola_window = librosa.util.normalize(hann, norm=None) ** 2
        ola_window = torch.Tensor(ola_window)
        self.register_buffer("ola_window", ola_window)

        if freeze_parameter:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, real_stft, imag_stft, length=None):
        """
        shape should be (bs, nfft//2+1, nframes)
        """
        assert real_stft.dim() == 3 and imag_stft.dim() == 3
        bs, _, nframes = real_stft.shape

        full_real = torch.cat(
            (real_stft, torch.flip(real_stft[:, 1:-1, :], dims=[1])), dim=1
        )
        full_imag = torch.cat(
            (imag_stft, -torch.flip(imag_stft[:, 1:-1, :], dims=[1])), dim=1
        )
        # (bs, nfft, nframes)
        # print(self.conv_real.weight.data[10:20, 10:20, 0])
        # print(self.conv_imag.weight.data[10:20, 10:20, 0])
        # print(full_real[0, 10:20, 10:20])
        # print(full_imag[0, 10:20, 10:20])

        s_real = self.conv_real(full_real) - self.conv_imag(full_imag)
        # print(self.conv_real)
        # print(self.conv_imag)
        # print(s_real)

        # overlap
        output_samples = (nframes - 1) * self.hop_size + self.nfft
        y = F.fold(
            input=s_real,
            output_size=(1, output_samples),
            kernel_size=(1, self.nfft),
            stride=(1, self.hop_size),
        )  # (bs, 1, 1, audio_samples)
        y = y.squeeze()

        window_matrix = self.ola_window[None, :, None].repeat(1, 1, nframes)
        ifft_window_sum = F.fold(
            input=window_matrix,
            output_size=(1, output_samples),
            kernel_size=(1, self.nfft),
            stride=(1, self.hop_size),
        )  # (1, 1, 1, audio_samples)
        ifft_window_sum = ifft_window_sum.squeeze()
        ifft_window_sum = ifft_window_sum.clamp(1e-11, np.inf)

        y = y / ifft_window_sum[None, :]
        if length:
            y = y[:, self.nfft // 2 : self.nfft // 2 + length]
        else:
            y = y[:, self.nfft // 2 : -self.nfft // 2]

        return y

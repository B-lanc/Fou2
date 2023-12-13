import torch
import torch.nn as nn
import lightning as L
from torch.optim.optimizer import Optimizer

from .modules import EMA, STFT, ISTFT, FFC, ConvBlock, FFT

import random


class Fou2Model(nn.Module):
    def __init__(
        self, levels, channel, depth, chgn, dropout, freeze_parameters, width, height
    ):
        super(Fou2Model, self).__init__()

        self.fft_w = FFT(width, freeze_parameters)
        self.fft_h = FFT(height, freeze_parameters)
        self.conv_in = ConvBlock(2, channel, chgn, dropout, 1, 1, 0)

        self.blocks = nn.ModuleList()
        for _ in range(levels):
            bl = nn.ModuleList()
            for _ in range(depth):
                bl.append(FFC(channel, channel, dropout, self.fft_w, self.fft_h))
            self.blocks.append(bl)
        self.conv_out = nn.Conv2d(channel, 2, 1, 1, 0)

    def forward(self, x):
        x = self.conv_in(x)

        xg = torch.zeros_like(x)
        for fus in self.blocks:
            for fu in fus:
                yl, yg = fu(x, xg)
            x = x + yl
            xg = xg + yg

        x = self.conv_out(x)
        return x


class Fou2(L.LightningModule):
    def __init__(self, model_cfg, ema_cfg, stft_cfg, lr):
        super(Fou2, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = Fou2Model(
            model_cfg.level,
            model_cfg.channel,
            model_cfg.depth,
            model_cfg.gn_channels,
            model_cfg.dropout,
            stft_cfg.freeze_parameters,
            model_cfg.width,
            stft_cfg.nfft // 2,
        )
        self.ema_model = EMA(self.model, ema_cfg.beta, ema_cfg.step_start)

        pad = stft_cfg.nfft // stft_cfg.hop_size if stft_cfg.padding else 0
        self.io = int((model_cfg.width - 1 - pad) * stft_cfg.hop_size + stft_cfg.nfft)

        self.stft = STFT(
            stft_cfg.nfft,
            stft_cfg.hop_size,
            stft_cfg.freeze_parameters,
            stft_cfg.padding,
            stft_cfg.pad_mode,
            stft_cfg.scaled,
        )
        self.istft = ISTFT(
            stft_cfg.nfft,
            stft_cfg.hop_size,
            stft_cfg.freeze_parameters,
            stft_cfg.scaled,
        )

        self.crit = nn.L1Loss()

    def forward(self, x, ema=False):
        r, i = self.stft(x)
        r = r[:, None, :, :]
        i = i[:, None, :, :]
        out = torch.cat((r, i), dim=1)  # (bs, 2, nfft//2+1, nframes)

        MODEL = self.ema_model if ema else self.model
        out = MODEL(out)

        r, i = out[:, 0, :, :], out[:, 1, :, :]
        audio = self.istft(r, i)

        return audio

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = self.crit(y, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = self.crit(y, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.ema_model.update_model(self.model, self.global_step)

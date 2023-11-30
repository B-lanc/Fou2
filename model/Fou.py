import torch
import lightning as L
from torch.optim.optimizer import Optimizer

from .modules import EMA, MaskingUNet, STFT, ISTFT, FFT, RMSE_Loss

import random


class Fou(L.LightningModule):
    def __init__(self, model_cfg, ema_cfg, stft_cfg, lr):
        super(Fou, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = MaskingUNet(
            model_cfg.in_channels,
            model_cfg.out_channels,
            model_cfg.channels,
            model_cfg.depth,
            model_cfg.gn_channels,
            model_cfg.dropout,
            model_cfg.attention_channels,
        )
        self.ema_model = EMA(self.model, ema_cfg.beta, ema_cfg.step_start)

        pad = stft_cfg.nfft // stft_cfg.hop_size if stft_cfg.padding else 0
        self.io_length = int(
            (model_cfg.width - 1 - pad) * stft_cfg.hop_size + stft_cfg.nfft
        )

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
        self.fft_w = FFT(
            model_cfg.width,
            stft_cfg.freeze_parameters,
        )
        self.fft_h = FFT(stft_cfg.nfft // 2, stft_cfg.freeze_parameters)
        self.crit = RMSE_Loss(1e-8)

    def forward(self, x, top):
        """
        top is whether to cut the last or first frequency index,
          if True, then the first frequency bin will be kept and last discarded
          if false, the first frequency bin will be discarded and the last kept
        discarded bin will use the original unprocessed bin

        for validation, will be run with both
        """

        r, i = self.stft(x)

        ax = -1 if top else 0
        _r = r[:, [ax], :]
        _i = i[:, [ax], :]
        r = r[:, :-1, :] if top else r[:, 1:, :]
        i = i[:, :-1, :] if top else i[:, 1:, :]

        r, i = self.fft_w(r, i, False)
        r = r.permute(0, 2, 1)
        i = i.permute(0, 2, 1)
        r, i = self.fft_h(r, i, False)

        r = r[:, None, :, :]
        i = i[:, None, :, :]
        _x = torch.cat((r, i), dim=1)

        _x = self.model(_x)
        r, i = _x[:, 0, :, :], _x[:, 1, :, :]

        r, i = self.fft_h(r, i, True)
        r = r.permute(0, 2, 1)
        i = i.permute(0, 2, 1)
        r, i = self.fft_w(r, i, True)

        r = (r, _r) if top else (_r, r)
        i = (i, _i) if top else (_i, i)
        r = torch.cat(r, dim=1)
        i = torch.cat(i, dim=1)
        _x = self.istft(r, i)

        return _x

    def both(self, x, ema=False):
        """
        doing forward, but with both top and bottom, and using both
        """
        r, i = self.stft(x)

        r1 = r[:, :-1, :]
        r2 = r[:, 1:, :]
        i1 = i[:, :-1, :]
        i2 = i[:, 1:, :]

        r1, i1 = self.fft_w(r1, i1, False)
        r2, i2 = self.fft_w(r2, i2, False)
        r1 = r1.permute(0, 2, 1)
        i1 = i1.permute(0, 2, 1)
        r2 = r2.permute(0, 2, 1)
        i2 = i2.permute(0, 2, 1)
        r1, i1 = self.fft_h(r1, i1, False)
        r2, i2 = self.fft_h(r2, i2, False)

        r1 = r1[:, None, :, :]
        i1 = i1[:, None, :, :]
        r2 = r2[:, None, :, :]
        i2 = i2[:, None, :, :]
        _x1 = torch.cat((r1, i1), dim=1)
        _x2 = torch.cat((r2, i2), dim=1)

        MODEL = self.ema_model if ema else self.model
        _x1 = MODEL(_x1)
        _x2 = MODEL(_x2)
        r1, i1 = _x1[:, 0, :, :], _x1[:, 1, :, :]
        r2, i2 = _x2[:, 0, :, :], _x2[:, 1, :, :]

        r1, i1 = self.fft_h(r1, i1, True)
        r2, i2 = self.fft_h(r2, i2, True)
        r1 = r1.permute(0, 2, 1)
        i1 = i1.permute(0, 2, 1)
        r2 = r2.permute(0, 2, 1)
        i2 = i2.permute(0, 2, 1)
        r1, i1 = self.fft_w(r1, i1, True)
        r2, i2 = self.fft_w(r2, i2, True)

        ax = r1.shape[1] // 2
        r = (r1[:, : ax + 1], r2[:, ax:])
        i = (i1[:, : ax + 1], i2[:, ax:])
        r = torch.cat(r, dim=1)
        i = torch.cat(i, dim=1)
        _x = self.istft(r, i)

        return _x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        x, target = batch
        top = random.random() > 0.5
        y = self(x, top)
        loss = self.crit(y, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = self.both(x)
        loss = self.crit(y, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.ema_model.update_model(self.model, self.global_step)

import torch
import lightning as L
from torch.optim.optimizer import Optimizer

from .modules import EMA, MaskingUNet, STFT, ISTFT, FFT


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

        # _x = self.model(_x)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        pass

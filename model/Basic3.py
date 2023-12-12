import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from .modules import STFT, ISTFT, EMA
from .modules.Blocks import ConvBlock

crit = F.l1_loss


class ResU(nn.Module):
    def __init__(
        self, channels, depth, resu_channels, kernel_size, stride, chgn=8, dropout=0.2
    ):
        super(ResU, self).__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv_in = nn.Conv2d(2, channels, 3, 1, 1)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = []
            chin = channels
            for i in resu_channels:
                block.append(ConvBlock(chin, i, chgn, dropout, stride, stride, 0))
                block.append(ConvBlock(i, i, chgn, dropout, kernel_size, 1, padding))
                chin = i
            for i in range(len(resu_channels) - 1, -1, -1):
                if i == 0:
                    block.append(
                        nn.ConvTranspose2d(
                            resu_channels[0], channels, stride, stride, 0
                        )
                    )
                    block.append(
                        ConvBlock(
                            channels, channels, chgn, dropout, kernel_size, 1, padding
                        )
                    )
                else:
                    block.append(
                        nn.ConvTranspose2d(
                            resu_channels[i],
                            resu_channels[i - 1],
                            stride,
                            stride,
                            0,
                        )
                    )
                    block.append(
                        ConvBlock(
                            resu_channels[i - 1],
                            resu_channels[i - 1],
                            chgn,
                            dropout,
                            kernel_size,
                            1,
                            padding,
                        )
                    )

            self.blocks.append(nn.Sequential(*block))
        self.conv_out = nn.Conv2d(channels, 2, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            y = block(x)
            x = x + y
        x = self.conv_out(x)
        return x


class BasicModel3(L.LightningModule):
    def __init__(self, model_cfg, ema_cfg, stft_cfg, lr):
        super(BasicModel3, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = ResU(
            model_cfg.channels,
            model_cfg.depth,
            model_cfg.resu_channels,
            model_cfg.kernel_size,
            model_cfg.stride,
            model_cfg.chgn,
            model_cfg.dropout,
        )
        self.ema_model = EMA(self.model, ema_cfg.beta, ema_cfg.step_start)

        pad = stft_cfg.nfft // stft_cfg.hop_size if stft_cfg.padding else 0
        self.io_length = int(
            (model_cfg.io - 1 - pad) * stft_cfg.hop_size + stft_cfg.nfft
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

    def forward(self, x, ema=False):
        r, i = self.stft(x)
        r = r[:, None, :, :]
        i = i[:, None, :, :]
        out = torch.cat((r, i), dim=1)  # (bs, 2, nfft//2+1, nframes)
        botout = out[:, :, :256, :]
        topout = out[:, :, 1:, :]

        MODEL = self.ema_model if ema else self.model
        botout = MODEL(botout)
        topout = MODEL(topout)

        n = topout.shape[2] // 2
        out = torch.cat((botout[:, :, :n, :], topout[:, :, n - 1 :, :]), dim=2)

        r, i = out[:, 0, :, :], out[:, 1, :, :]
        audio = self.istft(r, i)

        return audio

    def get_io(self):
        return self.io_length

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = crit(y, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = crit(y, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        self.ema_model.update_model(self.model, self.global_step)

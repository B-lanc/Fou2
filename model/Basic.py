import torch
import torch.nn.functional as F
import lightning as L

from .modules import STFT, ISTFT, EMA, UNet

# TODO... possibly fix slight jank, especially with the cuttop/cutbot, but honestly it works fine
class BasicModel(L.LightningModule):
    """
    This model is kinda fucked.... since it's using UNet, the size of the inputs and outputs aren't always supported
    this is a big deal because of the STFT frequency bins, as it's size is "predetermined"
    I'll just test to see if cutting and padding would ruin the performance...
    """

    def __init__(self, model_cfg, ema_cfg, stft_cfg, lr):
        super(BasicModel, self).__init__()

        self.lr = lr
        self.model = UNet(
            model_cfg.in_channels,
            model_cfg.out_channels,
            model_cfg.channels,
            model_cfg.depth,
            model_cfg.kernel_size,
            model_cfg.padding,
            model_cfg.stride,
            model_cfg.dilation,
            model_cfg.gn_channels,
            model_cfg.dropout,
            model_cfg.attention_channels,
            model_cfg.Masking,
        )
        self.ema_model = EMA(self.model, ema_cfg.beta, ema_cfg.step_start)

        (self.ih, self.oh, _), (self.iw, self.ow, _) = self.model.get_io(
            model_cfg.min_output_height, model_cfg.min_output_width
        )
        pad = stft_cfg.nfft // stft_cfg.hop_size if stft_cfg.padding else 0
        self.input_length = int((self.iw - 1 - pad) * stft_cfg.hop_size + stft_cfg.nfft)
        self.output_length = int(
            (self.ow - 1 - pad) * stft_cfg.hop_size + stft_cfg.nfft
        )

        pad_ = stft_cfg.nfft // 2 + 1
        assert self.ih >= pad_  # so I can pad
        assert self.oh >= pad_  # so I can cut

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

        even = (self.ih - pad_) % 2 == 0
        self.padtop = (self.ih - pad_) // 2
        self.padbot = self.padtop if even else self.padtop + 1

        even = (self.oh - pad_) % 2 == 0
        self.cuttop = int((self.oh - pad_) // 2)
        self.cutbot = self.cuttop if even else self.cuttop + 1
        self.cuttop = None if self.cuttop == 0 else -self.cuttop

    def forward(self, x):
        r, i = self.stft(x)
        r = r[:, None, :, :]
        i = i[:, None, :, :]
        out = torch.cat((r, i), dim=1)  # (bs, 2, nfft//2+1, nframes)
        assert out.shape[-1] == self.iw
        out = F.pad(out, (0, 0, self.padbot, self.padtop), mode="reflect")
        assert out.shape[-2] == self.ih

        out = self.model(out)
        out = out[:, :, self.cutbot : self.cuttop, :]
        r, i = out[:, 0, :, :], out[:, 1, :, :]
        audio = self.istft(r, i)

        return audio

    def get_io(self):
        return self.input_length, self.output_length

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = F.mse_loss(y, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = F.mse_loss(y, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        self.ema_model.update_model(self.model, self.global_step)

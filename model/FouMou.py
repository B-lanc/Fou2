import torch
import torch.nn as nn
import lightning as L

from .modules import STFT, ISTFT, EMA


crit = torch.nn.functional.l1_loss


class CBAD(nn.Module):
    def __init__(self, dim, trans, chin, chout, kern, stride, pad, chgn, drop):
        super(CBAD, self).__init__()
        if dim == 1:
            CONV = nn.ConvTranspose1d if trans else nn.Conv1d
        elif dim == 2:
            CONV = nn.ConvTranspose2d if trans else nn.Conv2d
        else:
            raise ValueError("dim has to be either 1 or 2")

        self.conv = CONV(chin, chout, kern, stride, pad)
        self.gn = nn.GroupNorm(chgn, chout)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.dropout(self.act(self.gn(self.conv(x))))


class ATT(nn.Module):
    def __init__(self, ch, n_heads, drop):
        super(ATT, self).__init__()
        self.att = nn.MultiheadAttention(ch, n_heads, drop, batch_first=True)
        self.ff = nn.Linear(ch, ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = x.reshape(bs, h * w, ch)
        _x, _ = self.att(x, x, x)
        x = x + _x
        x = self.dropout(self.act(self.ff(x)))
        x = x.reshape(bs, ch, h, w)
        return x


class FoUnet(nn.Module):
    def __init__(self, cfg):
        super(FoUnet, self).__init__()

        chgn = cfg.chgn
        drop = cfg.dropout
        kern = cfg.kernel_size
        pad = kern // 2
        stride = cfg.stride

        self.conv_in = CBAD(2, False, 2, cfg.channels[0], 1, 1, 0, 1, drop)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.conv_out = CBAD(2, False, cfg.channels[0], 2, 1, 1, 0, 1, drop)

        chin = cfg.channels[0]
        for ch in cfg.channels:
            enc = []
            dec = []
            enc.append(CBAD(2, False, chin, ch, 1, 1, 0, chgn, drop))
            enc.append(CBAD(2, False, ch, ch, kern, stride, pad, chgn, drop))
            dec.append(CBAD(2, True, ch, ch, kern, stride, pad, chgn, drop))
            dec.append(CBAD(2, False, ch, chin, 1, 1, 0, chgn, drop))
            for _ in range(cfg.depth):
                enc.append(CBAD(2, False, ch, ch, kern, 1, pad, chgn, drop))
                dec.append(CBAD(2, False, chin, chin, kern, 1, pad, chgn, drop))

            self.encoder.append(nn.Sequential(*enc))
            self.decoder.insert(0, nn.Sequential(*dec))

            chin = ch

        self.bottle = nn.ModuleList()
        ch = cfg.channels[-1]
        for _ in range(cfg.att_depth):
            self.bottle.append(ATT(ch, cfg.n_heads, drop))

    def forward(self, z):
        z = self.conv_in(z)

        short = [z]
        for enc in self.encoder:
            z = enc(z)
            short.append(z)
        short.reverse()

        for att in self.bottle:
            z = att(z)

        for i in range(len(self.decoder)):
            z = z * short[i]
            z = self.decoder[i](z)

        z = z * short[-1]
        z = self.conv_out(z)
        return z


class Foumou(L.LightningModule):
    def __init__(self, cfg, ema_cfg, lr):
        super(Foumou, self).__init__()
        self.save_hyperparameters()
        hop_size = cfg.freq - 1
        nfft = hop_size * 2

        self.io = hop_size ** 2
        self.lr = lr

        self.stft = STFT(nfft, hop_size, True, True)
        self.istft = ISTFT(nfft, hop_size, True)

        self.model = FoUnet(cfg)
        self.ema_model = EMA(self.model, ema_cfg.beta, ema_cfg.step_start)

    def get_io(self):
        return self.io

    def forward(self, x, ema=False):
        r, i = self.stft(x)
        r = r[:, None, :, :]
        i = i[:, None, :, :]
        z = torch.cat((r, i), dim=1)  # (bs, 2, nfft//2+1, nframes)

        MODEL = self.ema_model if ema else self.model
        z = MODEL(z)

        r, i = z[:, 0, :, :], z[:, 1, :, :]
        z = self.istft(r, i)

        return z

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

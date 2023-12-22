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

    def forward(self, q, k=None):
        if k is None:
            k = q
        _q, _ = self.att(q, k, k)
        q = q + _q
        q = self.dropout(self.act(self.ff(q)))
        return q


class FoUnet(nn.Module):
    def __init__(self, cfg):
        super(FoUnet, self).__init__()

        chgn = cfg.chgn
        drop = cfg.dropout
        kern = cfg.kernel_size
        pad = kern // 2
        stride = cfg.stride

        self.z_in = CBAD(2, False, 2, cfg.channels[0], 1, 1, 0, 1, drop)
        self.z_encoder = nn.ModuleList()
        self.z_decoder = nn.ModuleList()
        self.z_out = CBAD(2, False, cfg.channels[0], 2, 1, 1, 0, 1, drop)

        self.t_in = CBAD(1, False, 1, cfg.channels[0], 1, 1, 0, 1, drop)
        self.t_encoder = nn.ModuleList()
        self.t_decoder = nn.ModuleList()
        self.t_out = CBAD(1, False, cfg.channels[0], 1, 1, 1, 0, 1, drop)

        chin = cfg.channels[0]
        for ch in cfg.channels:
            _zenc = []
            _tenc = []
            _zenc.append(CBAD(2, False, chin, ch, 1, 1, 0, chgn, drop))
            _zenc.append(CBAD(2, False, ch, ch, kern, stride, pad, chgn, drop))
            _tenc.append(CBAD(1, False, chin, ch, 1, 1, 0, chgn, drop))
            _tenc.append(CBAD(1, False, ch, ch, stride, stride, 0, chgn, drop))
            for _ in range(cfg.depth):
                _zenc.append(CBAD(2, False, ch, ch, kern, 1, pad, chgn, drop))
                _tenc.append(CBAD(1, False, ch, ch, kern, 1, pad, chgn, drop))
            self.z_encoder.append(nn.Sequential(*_zenc))
            self.t_encoder.append(nn.Sequential(*_tenc))

            _zdec = []
            _tdec = []
            _zdec.append(CBAD(2, True, ch, ch, kern, stride, pad, chgn, drop))
            _zdec.append(CBAD(2, False, ch, chin, 1, 1, 0, chgn, drop))
            _tdec.append(CBAD(1, True, ch, ch, stride, stride, 0, chgn, drop))
            _tdec.append(CBAD(1, False, ch, chin, 1, 1, 0, chgn, drop))
            for _ in range(cfg.depth):
                _zdec.append(CBAD(2, False, chin, chin, kern, 1, pad, chgn, drop))
                _tdec.append(CBAD(1, False, chin, chin, kern, 1, pad, chgn, drop))
            self.z_decoder.insert(0, nn.Sequential(*_zdec))
            self.t_decoder.insert(0, nn.Sequential(*_tdec))

            chin = ch

        self.z_self = nn.ModuleList()
        self.z_cross = nn.ModuleList()
        self.t_self = nn.ModuleList()
        self.t_cross = nn.ModuleList()
        ch = cfg.channels[-1]
        for _ in range(cfg.bottle_depth):
            self.z_self.append(ATT(ch, cfg.n_heads, drop))
            self.t_self.append(ATT(ch, cfg.n_heads, drop))
            self.z_cross.append(ATT(ch, cfg.n_heads, drop))
            self.t_cross.append(ATT(ch, cfg.n_heads, drop))

    def forward(self, z, t):
        z = self.z_in(z)
        t = self.t_in(t)

        zshort = [z]
        tshort = [t]
        for zenc, tenc in zip(self.z_encoder, self.t_encoder):
            z = zenc(z)
            t = tenc(t)
            zshort.append(z)
            tshort.append(t)
        zshort.reverse()
        tshort.reverse()

        bs, ch, zh, zw = z.shape
        _, _, tw = t.shape
        z = z.reshape((bs, zh * zw, ch))
        t = t.reshape((bs, tw, ch))
        for i in range(len(self.z_self)):
            _z = self.z_self[i](z, None)
            _t = self.t_self[i](t, None)
            z = self.z_cross[i](_z, _t)
            t = self.t_cross[i](_t, _z)
        z = z.reshape((bs, ch, zh, zw))
        t = t.reshape((bs, ch, tw))

        for i in range(len(self.z_decoder)):
            z = z * zshort[i]
            t = t * tshort[i]
            z = self.z_decoder[i](z)
            t = self.t_decoder[i](t)

        z = z * zshort[-1]
        t = t * tshort[-1]
        z = self.z_out(z)
        t = self.t_out(t)
        return z, t


class Fouivre(L.LightningModule):
    def __init__(self, cfg, ema_cfg, lr):
        super(Fouivre, self).__init__()
        hop_size = cfg.freq - 1
        nfft = hop_size * 2

        self.io = hop_size ** 2
        self.lr = lr

        self.stft = STFT(nfft, hop_size, True, True)
        self.itft = ISTFT(nfft, hop_size, True)

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
        z, t = MODEL(z, x)

        r, i = z[:, 0, :, :], z[:, 1, :, :]
        z = self.istft(r, i)

        return z + t

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

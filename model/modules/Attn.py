import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, channels, gn_channels=8):
        super(AttentionBlock, self).__init__()
        assert channels % gn_channels == 0

        self.norm = nn.GroupNorm(gn_channels, channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = x.shape

        q = q.reshape(b, c, w * h)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, w * h)

        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** 0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, w * h)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.out(h_)

        return x + h

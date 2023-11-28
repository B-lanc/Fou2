import torch.nn as nn

from .Blocks import ConvBlock
from .Attn import AttentionBlock

from functools import partial


class MaskingUNet(nn.Module):
    """
    MaskingUNet applies an element-wise multiplication (as if a mask) instead of the original UNet idea of concatenating the shortcut
    Only works with data in the form of (bs, ch, h, w), where the h and w is divisible by 2^(len(channels)-1)
    usually just use powers of 2 as h and w like 256x256
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        channels,
        depth=2,
        gn_channels=8,
        dropout=0.1,
        attention_channels=[],
    ):
        for ch in channels:
            assert ch % gn_channels == 0
        ATT = False
        if len(attention_channels) > 0 or attention_channels is not None:
            assert len(attention_channels) == (len(channels) - 1)
            ATT = True
        assert len(channels) > 1

        super(MaskingUNet, self).__init__()

        CONV = partial(
            ConvBlock,
            chgn=gn_channels,
            dropout=dropout,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        self.depth = depth
        self.down_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        self.down_ch = nn.ModuleList()
        self.up_ch = []
        self.up = []
        self.up_blocks = []

        self.pre = nn.Conv2d(in_channels, channels[0], 1, 1, 0)
        self.post = nn.Conv2d(channels[0], out_channels, 1, 1, 0)
        for idx in range(len(channels) - 1):
            down_blocks = []
            up_blocks = []
            ch = channels[idx]
            for _ in range(depth):
                down_blocks.append(CONV(ch, ch))
                up_blocks.append(CONV(ch, ch))
                if ATT:
                    if attention_channels[idx]:
                        down_blocks.append(AttentionBlock(ch, gn_channels))
                        up_blocks.append(AttentionBlock(ch, gn_channels))
            self.down_blocks.append(nn.Sequential(*down_blocks))
            self.down.append(nn.Conv2d(ch, ch, 2, 2, 0))
            self.down_ch.append(CONV(ch, channels[idx + 1]))
            self.up_ch.append(CONV(channels[idx + 1], ch))
            self.up.append(nn.ConvTranspose2d(ch, ch, 2, 2, 0, 0))
            self.up_blocks.append(nn.Sequential(*up_blocks))

        self.up_ch = nn.ModuleList(self.up_ch[::-1])
        self.up = nn.ModuleList(self.up[::-1])
        self.up_blocks = nn.ModuleList(self.up_blocks[::-1])

        self.bottle = nn.Sequential(
            *[CONV(channels[-1], channels[-1]) for _ in range(depth)],
        )

    def forward(self, x):
        x = self.pre(x)
        short = []
        for block, down, ch in zip(self.down_blocks, self.down, self.down_ch):
            short.append(x)
            x = block(x)
            x = down(x)
            x = ch(x)

        x = self.bottle(x)
        short.reverse()

        for up, block, ch, sh in zip(self.up, self.up_blocks, self.up_ch, short):
            x = ch(x)
            x = up(x)
            x = block(x)
            x = sh * x

        x = self.post(x)
        return x

import torch
import torch.nn as nn

from Blocks import ConvBlock
from Attn import AttentionBlock
from utils import crop_like

from functools import partial


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels,
        depth=2,
        kernel_size=3,
        padding=0,
        stride=2,
        dilation=1,
        gn_channels=8,
        dropout=0.1,
        attention_channels=[],
        Masking=False,
    ):
        """
        Channels should be

        attention_channels should either be an empty list, None, or a list of booleans or 1 and 0s, where True would mean there should be an attention block there, the size should be 1 less than channels
        example, if channels is [32, 64, 128], attention_channels can be [False, True]
        """
        for ch in channels:
            assert ch % gn_channels == 0
        ATT = False
        if len(attention_channels) > 0 or attention_channels is not None:
            assert len(attention_channels) == (len(channels) - 1)
            ATT = True
        assert len(channels) > 1

        super(UNet, self).__init__()

        CONV = partial(
            ConvBlock,
            chgn=gn_channels,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )

        self.Masking = Masking
        self.down_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = []
        self.up_blocks = []

        chin = in_channels
        chout = channels[0]
        for idx in range(len(channels - 1)):
            down_blocks = []
            up_blocks = []
            chout = channels[idx]
            upchin = chout if Masking else chout * 2
            for _ in range(depth):
                down_blocks.append(CONV(chin, chout))
                up_blocks.append(CONV(upchin, chout))
                if ATT:
                    if attention_channels[idx]:
                        down_blocks.append(AttentionBlock(chout, gn_channels))
                        up_blocks.append(AttentionBlock(chout, gn_channels))
                chin = chout
                upchin = chout
            self.down_blocks.append(nn.Sequential(*down_blocks))
            self.down.append(nn.Conv2d(chout, chout, kernel_size, stride, padding, 1))
            self.up.append(
                nn.ConvTranspose2d(
                    channels[idx + 1], chout, kernel_size, stride, padding, 1
                )
            )
            self.up_blocks.append(nn.Sequential(*up_blocks))
        self.up = nn.ModuleList(self.up[::-1])
        self.up_blocks = nn.ModuleList(self.up_blocks[::-1])

        self.bottle = nn.Sequential(
            CONV(channels[-2], channels[-1]),
            CONV(
                channels[-1],
                channels[-2],
            ),
        )
        self.post = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, x):
        short = []
        for block, down in zip(self.down_blocks, self.down):
            x = block(x)
            short.append(x)
            x = down(x)

        x = self.bottle(x)
        short.reverse()

        for up, block, sh in zip(self.up, self.up_blocks, short):
            x = up(x)
            sh = crop_like(sh, x)
            if self.Masking:
                x = sh * x
            else:
                x = torch.cat((x, sh), dim=1)

        x = self.post(x)
        return x

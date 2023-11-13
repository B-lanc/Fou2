import torch
import torch.nn as nn

from .Blocks import ConvBlock
from .Attn import AttentionBlock
from .utils import crop_like, get_conv_input, get_conv_output

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
        self.depth = depth
        self.down_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        self.down_ch = nn.ModuleList()
        self.up_ch = []
        self.up = []
        self.up_blocks = []

        self.pre = nn.Conv2d(in_channels, channels[0], 1, 1, 0)
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
            self.down.append(nn.Conv2d(ch, ch, kernel_size, stride, 0))
            self.down_ch.append(CONV(ch, channels[idx + 1]))
            chin = (
                channels[idx + 1]
                if Masking or idx == len(channels) - 2
                else channels[idx + 1] * 2
            )
            self.up_ch.append(CONV(chin, ch))
            self.up.append(nn.ConvTranspose2d(ch, ch, kernel_size, stride, 0, 0))
            self.up_blocks.append(nn.Sequential(*up_blocks))

        self.up_ch = nn.ModuleList(self.up_ch[::-1])
        self.up = nn.ModuleList(self.up[::-1])
        self.up_blocks = nn.ModuleList(self.up_blocks[::-1])

        self.bottle = nn.Sequential(
            CONV(channels[-1], channels[-1]),
            CONV(
                channels[-1],
                channels[-1],
            ),
        )
        chin = channels[0] if Masking else channels[0] * 2
        self.post = nn.Conv2d(chin, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.pre(x)
        short = []
        for block, down, ch in zip(self.down_blocks, self.down, self.down_ch):
            x = block(x)
            short.append(x)
            x = down(x)
            x = ch(x)

        x = self.bottle(x)
        short.reverse()

        for up, block, ch, sh in zip(self.up, self.up_blocks, self.up_ch, short):
            x = ch(x)
            x = up(x)
            x = block(x)
            sh = crop_like(sh, x)
            if self.Masking:
                x = sh * x
            else:
                x = torch.cat((x, sh), dim=1)

        x = self.post(x)
        return x

    def get_io(self, min_output_height, min_output_width):
        # THIS IS SO JANK GOD.....
        bh, bw = 1, 1
        height = []
        width = []
        while len(height) == 0 or len(width) == 0:
            hflag = True
            wflag = True
            oh, ow = self.bottle[1].get_output(bh, bw)
            ih, iw = self.bottle[0].get_input(bh, bw)
            for d_b, d, d_c, u_c, u, u_b in zip(
                self.down_blocks[::-1],
                self.down[::-1],
                self.down_ch[::-1],
                self.up_ch,
                self.up,
                self.up_blocks,
            ):
                oh, ow = u_c.get_output(oh, ow)
                oh, ow = get_conv_input(u, oh, ow)
                ih, iw = d_c.get_input(ih, iw)
                ih, iw = get_conv_input(d, ih, iw)

                shorth, shortw = ih, iw

                for _ in range(self.depth):
                    oh, ow = u_b[0].get_output(oh, ow)
                    ih, iw = d_b[0].get_input(ih, iw)

                if (shorth - oh) % 2 != 0:
                    hflag = False
                if (shortw - ow) % 2 != 0:
                    wflag = False
                if not (hflag or wflag):
                    break

            if hflag and len(height) == 0 and oh >= min_output_height:
                height.append(ih)
                height.append(oh)
                height.append(bh)
            if wflag and len(width) == 0 and ow >= min_output_width:
                width.append(iw)
                width.append(ow)
                width.append(bw)

            bh += 1
            bw += 1

        return height, width

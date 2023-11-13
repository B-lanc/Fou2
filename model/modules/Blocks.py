import torch.nn as nn

from .utils import get_conv_input, get_conv_output


class ConvBlock(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        chgn,
        dropout,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
    ):
        assert chout % chgn == 0

        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(chin, chout, kernel_size, stride, padding, dilation),
            nn.GroupNorm(chgn, chout),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)

    def get_input(self, h, w):
        return get_conv_input(self.block[0], h, w)

    def get_output(self, h, w):
        return get_conv_output(self.block[0], h, w)

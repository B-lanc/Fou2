import torch
import torch.nn as nn

from .utils import get_conv_input, get_conv_output


class DownConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        kernel_size=3,
        padding=0,
        stride=2,
        dilation=1,
        gn_channels=8,
        dropout=0.1,
    ):
        super(DownConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.depth = depth

        self.activation = nn.ReLU()
        self.downconv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding
        )
        self.downnorm = nn.GroupNorm(gn_channels, in_channels)
        convs = []
        for _ in range(depth):
            convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
            )
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = self.downconv(x)
        x = self.downnorm(x)
        x = self.activation(x)

        x = self.convs(x)
        return x

    def get_input_hw(self, height, width):
        h, w = height, width
        for _ in range(self.depth):
            h, w = get_conv_input(self.convs[0], h, w)
        h, w = get_conv_input(self.downconv, h, w)
        return h, w

    def get_output_hw(self, height, width):
        h, w = get_conv_output(self.downconv, height, width)
        for _ in range(self.depth):
            h, w = get_conv_output(self.convs[0], h, w)
        return h, w


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        kernel_size=3,
        padding=0,
        stride=2,
        dilation=1,
        gn_channels=8,
        dropout=0.1,
    ):
        super(UpConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.depth = depth

        self.activation = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.upnorm = nn.GroupNorm(gn_channels, out_channels)
        convs = []
        in_channels = out_channels * 2
        for _ in range(depth):
            convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation)
            )
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

    def forward(self, x, short):
        x = self.upconv(x)
        x = self.upnorm(x)
        x = self.activation(x)
        x = torch.cat((x, short), dim=1)

        x = self.convs(x)
        return x

    def get_input_hw(self, height, width):
        h, w = height, width
        for _ in range(self.depth):
            h, w = get_conv_input(self.convs[0], h, w)
        h, w = get_conv_output(self.upconv, h, w)
        return h, w

    def get_output_hw(self, height, width):
        h, w = get_conv_input(self.upconv, height, width)
        for _ in range(self.depth):
            h, w = get_conv_output(self.convs[0], h, w)
        return h, w


class MaskUpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        kernel_size=3,
        padding=0,
        stride=2,
        dilation=1,
        gn_channels=8,
        dropout=0.1,
    ):
        super(MaskUpConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.depth = depth

        self.activation = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.upnorm = nn.GroupNorm(gn_channels, out_channels)
        convs = []
        for _ in range(depth):
            convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, dilation)
            )
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, short):
        x = self.upconv(x)
        x = self.upnorm(x)
        x = self.activation(x)
        x = x * short

        x = self.convs(x)
        return x

    def get_input_hw(self, height, width):
        h, w = height, width
        for _ in range(self.depth):
            h, w = get_conv_input(self.convs[0], h, w)
        h, w = get_conv_output(self.upconv, h, w)
        return h, w

    def get_output_hw(self, height, width):
        h, w = get_conv_input(self.upconv, height, width)
        for _ in range(self.depth):
            h, w = get_conv_output(self.convs[0], h, w)
        return h, w

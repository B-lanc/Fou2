import torch

def get_conv_output(conv_block, height, width):
    k_h, k_w = conv_block.kernel_size
    s_h, s_w = conv_block.stride
    p_h, p_w = conv_block.padding
    d_h, d_w = conv_block.dilation

    out_h = (height + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1
    out_w = (width + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1
    return out_h, out_w


def get_conv_input(conv_block, height, width):
    k_h, k_w = conv_block.kernel_size
    s_h, s_w = conv_block.stride
    p_h, p_w = conv_block.padding
    d_h, d_w = conv_block.dilation

    in_h = (height - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1
    in_w = (width - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1
    return in_h, in_w


def crop_like(x, y):
    """
    shape = (bs, ch, h, w)
    y is smaller than x
    """
    h = x.shape[2] - y.shape[2]
    w = x.shape[3] - y.shape[3]
    assert h % 2 == 0
    assert w % 2 == 0
    h = h // 2
    w = w // 2
    hh = -h if h else None
    ww = -w if w else None
    return x[:, :, h:hh, w:ww]


def RMSE_Loss(eps):
    def _rmse(x, y):
        return torch.sqrt(eps + torch.nn.MSELoss(x, y))
import torch
import numpy as np

import math


def inference(model, song, input_size, output_size, batch_size=64):
    """
    song => (bs, length)
    """
    ch, orig_length = song.shape
    hop = output_size // 4 * 3
    N = (orig_length - output_size + hop) / hop
    N = int(math.ceil(N))
    HL = output_size + hop * (N - 1)

    diff = input_size - output_size
    pad_front = (diff // 2) + (diff % 2)
    pad_back = diff // 2 + HL - orig_length

    song = torch.Tensor(song)
    song = torch.nn.functional.pad(song, (pad_front, pad_back), mode="constant")

    X = torch.zeros((0, input_size))
    Y = np.zeros((0, output_size))
    for i in range(N):
        sl = i * hop
        X = torch.cat((X, song[:, sl : sl + input_size]), dim=0)

    times = int(math.ceil(N / batch_size * ch))

    X = X.to(model.device)
    for i in range(times):
        start = i * batch_size
        end = start + batch_size
        end = X.shape[0] if end > X.shape[0] else end
        _y = model(X[start:end]).detach().cpu().numpy()
        Y = np.concatenate((Y, _y), axis=0)

    out = np.zeros((ch, 0))
    start = (output_size - hop) // 2 + (output_size - hop) % 2
    end = (output_size - hop) // 2
    out = np.concatenate((out, Y[0:ch, :end]), axis=1)
    for i in range(1, Y.shape[0] // ch - 1):
        _y = Y[i * ch : i * ch + ch, start, end]
        out = np.concatenate((out, _y), axis=1)
    out = np.concatenate((out, Y[-ch:, start:]), axis=1)
    out = out[:, :orig_length]

    return out

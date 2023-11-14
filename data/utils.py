from scipy.signal import lfilter
import numpy as np


def all_pass_filter(signal, center_freq, Q, sr=44100):
    center_freq = center_freq / sr
    BW = center_freq / Q
    h_ = np.tan(np.pi * BW)
    c = (h_ - 1) + (h_ + 1)
    d = -np.cos(2 * np.pi * center_freq)

    b = [-c, d * (1 - c), 1]
    a = b[::-1]

    return lfilter(b, a, signal)

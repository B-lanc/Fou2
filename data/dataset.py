from torch.utils.data import Dataset
import h5py
import numpy as np
from sortedcontainers import SortedList

import random

from .utils import all_pass_filter


class AudioHDF(Dataset):
    """
    This dataset is for when having 2 hdf files, both of them just having indexes directly returning a single dimension numpy array
    f["1"] = single channel audio in numpy form

    1 hdf file is for the instrumentals/noise
    the other one is for the vocals

    main path is the one which will be used as length (epoch)
    sub path will be picked at random
    """

    def __init__(self, main_path, sub_path, input_length, output_length, random_hop):
        #TODO: decide what to do if input length and output length is the same, if it's not the same, and especially if the difference is odd
        self.main = main_path
        self.sub = sub_path
        self.input_length = input_length
        self.output_length = output_length
        self.diff = input_length - output_length
        self.random_hop = random_hop

        with h5py.File(self.main, "r") as f:
            lengths = [len(i) // output_length for i in f]
            self.main_lengths = SortedList(np.cumsum(lengths))
        with h5py.File(self.sub, "r") as f:
            lengths = [len(i) // output_length for i in f]
            self.sub_lengths = SortedList(np.cumsum(lengths))

    def __getitem__(self, idx):
        main_song_idx = self.main_lengths.bisect_left(idx)
        main_idx = idx - self.main_lengths[main_song_idx] if main_song_idx > 0 else idx

        if idx > self.sub_lengths[-1]:
            idx = int(random.random() * self.sub_lengths[-1])
        sub_song_idx = self.sub_lengths.bisect_left(idx)
        sub_idx = idx - self.sub_lengths[sub_song_idx] if sub_song_idx > 0 else idx

    def __len__(self):
        return self.main_lengths[-1]

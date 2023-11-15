import hydra
from omegaconf import DictConfig
import librosa
import h5py
import glob
import os
import numpy as np


def zero_index(audio, cons_zeros):
    iszero = np.concatenate(([0], np.equal(audio, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    idxs = (absdiff == 1).nonzero()[0].reshape(-1, 2)
    idxs = idxs[(idxs[:, 1] - idxs[:, 0]) >= cons_zeros]
    return idxs


def remove_zeros(audio, cons_zeros):
    if cons_zeros > 0:
        if audio.any():
            idxs = zero_index(audio, cons_zeros)
            for i in reversed(idxs):
                audio = np.delete(audio, np.s_[i[0] + 1 : i[1] + 1])
    return audio


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.hdf_dir):
        os.makedirs(cfg.hdf_dir)
    vox_hdf = os.path.join(cfg.hdf_dir, "vocals.hdf5")
    inst_hdf = os.path.join(cfg.hdf_dir, "instrumentals.hdf5")
    vox = os.path.join(cfg.dataset_dir, "vocals", "*")
    inst = os.path.join(cfg.dataset_dir, "instrumentals", "*")
    vox = glob.glob(vox)
    inst = glob.glob(inst)

    with h5py.File(vox_hdf, "w") as f:
        for idx, track in enumerate(vox):
            song, _ = librosa.load(track, mono=True, sr=cfg.prep.sr)
            song = remove_zeros(song, cfg.prep.cons_zeros)
            f.create_dataset(f"{idx}", data=song)
    with h5py.File(inst_hdf, "w") as f:
        for idx, track in enumerate(inst):
            song, _ = librosa.load(track, mono=True, sr=cfg.prep.sr)
            song = remove_zeros(song, cfg.prep.cons_zeros)
            f.create_dataset(f"{idx}", data=song)


if __name__ == "__main__":
    main()

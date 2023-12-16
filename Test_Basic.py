import torch
from model import BasicModel
import librosa
import soundfile as sf
from omegaconf import OmegaConf
from fastsdr import fastsdr
import numpy as np

from infer import inference
import argparse

import os
import glob


def main(args):
    inst_path = os.path.join(args.test_path, "instrumentals")
    vox_path = os.path.join(args.test_path, "vocals")
    song_names = glob.glob(os.path.join(inst_path, "*"))
    song_names = [os.path.basename(song) for song in song_names]

    device = "cuda" if args.cuda else "cpu"
    model = BasicModel.load_from_checkpoint(
        args.model_checkpoint_file,
    ).to(device)
    input_length, output_length = model.get_io()

    results = {}
    for song in song_names:
        print("Loading song :", song)
        vox, _ = librosa.load(os.path.join(vox_path, song), sr=args.sr, mono=args.mono)
        inst, _ = librosa.load(
            os.path.join(inst_path, song), sr=args.sr, mono=args.mono
        )
        audio = vox + inst

        with torch.no_grad():
            res = inference(
                model, audio, input_length, output_length, args.batch_size, args.ema
            )
        results[song] = fastsdr(inst.T[None, :, :], res.T[None, :, :])

    comb = 0
    testing = 0
    for song, res in results.items():
        meaned = np.nanmean(res)
        print(f"{song}: \t{meaned}")
        comb += meaned
        testing2 = testing
        testing = res
    comb = comb / len(results)
    print(comb)
    sep = np.concatenate(results.values(), axis=0)
    print(np.nanmean(sep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint_file")
    parser.add_argument("test_path")
    parser.add_argument("--sr", default=44100)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--batch_size", default=64)

    args = parser.parse_args()
    main(args)

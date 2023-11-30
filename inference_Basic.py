import torch
from model import BasicModel
import librosa
import soundfile as sf
from omegaconf import OmegaConf

from infer import inference
import argparse


def main(args, cfg):
    audio, _ = librosa.load(args.input_file, sr=args.sr, mono=args.mono)
    device = "cuda" if args.cuda else "cpu"
    model = BasicModel.load_from_checkpoint(
        args.model_checkpoint_file,
    ).to(device)
    input_length, output_length = model.get_io()

    with torch.no_grad():
        res = inference(model, audio, input_length, output_length, args.batch_size, args.ema)
    sf.write(args.output, res.T, args.sr, "PCM_24")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("model_checkpoint_file")
    parser.add_argument("-o", "--output", default="output.wav")
    parser.add_argument("--sr", default=44100)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--batch_size", default=64)

    args = parser.parse_args()
    cfg = OmegaConf.load("config.yaml")
    main(args, cfg)

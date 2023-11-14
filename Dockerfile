FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


RUN apt update && apt install -y \
    libsndfile1-dev \
    ffmpeg

RUN pip install librosa==0.10.0 \
    soundfile==0.12.1 \
    h5py==3.8.0 \
    pyloudnorm==0.1.1 \
    fastsdr \
    hydra-core=1.3.2 \
    lightning

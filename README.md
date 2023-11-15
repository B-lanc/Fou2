# Fou2
Fou mk.2

Building the docker image
> docker build . -t b-lanc/fou2

Running the docker container
> docker run -dit --name=fou --privileged --runtime=nvidia --gpus=all --shm-size=2gb -v /mnt/Data/datasets/fou/musdb18hq_restructured:/dataset -v /mnt/Data2/DockerVolumes/fou:/saves -v .:/workspace b-lanc/fou2

Going into the docker container
> docker exec -it thesis /bin/bash


Dataset format
dataset/
  train/
    instrumentals/
      filename1.wav
      filename2.wav
      ...
    vocals/
      filenamex.wav
      filenamey.wav
      ...
    hdf/
      instrumentals.hdf5
      vocals.hdf5
  test/
    instrumentals/
      filename1.wav
      filename2.wav
      ...
    vocals/
      filename1.wav
      filename2.wav
      ...
    hdf/
      instrumentals.hdf5
      vocals.hdf5

it's not enforced, but the test is recommended to be from the same track (something like in musdb18hq)
while for train, there's no real restriction
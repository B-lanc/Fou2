# Fou2
Fou mk.2

Building the docker image
> docker build . -t b-lanc/fou2

Running the docker container
> docker run -dit --name=fou --privileged --runtime=nvidia --gpus=all --shm-size=2gb -v /mnt/Data/datasets/fou/musdb18hq_restructured:/dataset -v /mnt/Data2/DockerVolumes/fou:/saves -v .:/workspace b-lanc/fou2

Going into the docker container
> docker exec -it thesis /bin/bash
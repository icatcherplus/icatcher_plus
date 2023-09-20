# Docker Usage

Provided [here](https://hub.docker.com/r/questengineering/icatcher_plus) is an automatically updated Docker image based on the latest iCatcher+ version available. This image acts as the instructions to create an easily runnable container with all of the tools necessary to execute iCatcher+. (Note: Performance running the container will be best on GPU, although it is possible to run on CPU.)

## 1. Download Docker

Docker can be downloaded [here](https://docs.docker.com/get-docker/). 

## 2. Get the container image

The iCatcher+ container image is easily pullable from [Docker Hub](https://hub.docker.com/r/questengineering/icatcher_plus). In order to pull the container image to your computer, run:

```bash
docker pull questengineering/icatcher_plus:latest
```

You can then run:

```bash
docker image ls
```

which should now show the container you downloaded.

## 3. Explore container

Running the following command:

```bash
docker run -it --rm=true questengineering/icatcher_plus:latest /bin/bash
```

will allow you to run a copy of the container. The container can be explored and iCatcher+ can be run easily this way. 


## 4. (Optional) Using Singularity

This Docker image can be easily pulled into Singularity by running:

```bash
singularity pull icatcher_plus.sif docker://questengineering/icatcher_plus:latest
```

## 5. Mounting your video directory

Assuming you have a local video/video directory that you'd like to run iCatcher+ on, you'll have to mount your local directory on the container. First you'll have to define your local directory as well as the path to where the directory will be mounted on your container:

```bash
# Set the path to your local video directory
LOCAL_VIDEO_DIR="/path/to/local/video/dir"

# Set the path where the local video directory will be mounted in the container
CONTAINER_VIDEO_DIR="/path/to/container/video/dir"
```

Next, you'll have to run your container in a writable mode, and mount the directory.

For Docker, this will look similar to:

```bash
docker run -v $LOCAL_VIDEO_DIR:$CONTAINER_VIDEO_DIR:rw my_docker_image
```

For Singularity, this will look similar to:

```bash
singularity exec --writable --bind $LOCAL_VIDEO_DIR:$CONTAINER_VIDEO_DIR my_singularity_image.sif my_command
```


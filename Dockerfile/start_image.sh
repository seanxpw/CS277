#!/bin/bash

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

PROJ="cs277"
DOCKERFILE_PATH="./"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME="${PROJ}_container"
SRC_DIR="${GITTOP}"
SRC_TARGET_DIR="/home"

if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

docker run -d \
    -it \
    --rm       \
    --gpus all \
    --privileged \
    --ulimit core=-1 \
    --name ${CONTAINER_NAME} \
    -u $(id -u $USER):$(id -g $USER) \
    --mount type=bind,source=${SRC_DIR},target=/home/CS277 \
    --mount type=bind,source=/trace,target=/home/trace,readonly \
    --mount type=bind,source=/usr/local/include,target=/usr/local/include,readonly \
    --mount type=bind,source=/usr/local/lib,target=/usr/local/lib,readonly \
    --mount type=bind,source=/nfshome/hoz006/IntelMKL,target=/home/IntelMKL \
    ${IMAGE_NAME} 
        #--mount type=bind,source=/nfshome/hoz006/IntelMKL,target=/home/IntelMKL \
        # --mount type=bind,source=/usr/local/cuda,target=/usr/local/cuda,readonly \
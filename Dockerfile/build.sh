#!/bin/bash

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

PROJ="cs277"
DOCKERFILE_PATH="./"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME="${PROJ}_container"
SRC_DIR="${GITTOP}"
SRC_TARGET_DIR="/home"

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

docker run -d \
    -it \
    --rm       \
    --gpus all \
    --name ${CONTAINER_NAME} \
    -u $(id -u $USER):$(id -g $USER) \
    --mount type=bind,source=${SRC_DIR},target=/home/CS277 \
    --mount type=bind,source=/usr/local/include,target=/usr/local/include,readonly \
    --mount type=bind,source=/usr/local/lib,target=/usr/local/lib,readonly \
    ${IMAGE_NAME} 
    
        # --mount type=bind,source=/usr/local/cuda,target=/usr/local/cuda,readonly \
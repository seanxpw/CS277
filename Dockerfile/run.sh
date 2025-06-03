#!/bin/bash

PROJ="cs277"
CONTAINER_NAME="${PROJ}_container"
docker exec -it --user root ${CONTAINER_NAME} bash
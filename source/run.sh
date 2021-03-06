#!/bin/bash

if [ $# -ne 1 ] ; then
  echo usage : 1 parameter is required
  exit 1
fi

if [[ ! -z  `which nvidia-docker`  ]]
then
    DOCKER_CMD=nvidia-docker
elif [[ ! -z  `which docker`  ]]
then
    echo "WARNING: nvidia-docker not found. Nvidia drivers may not work." >&2
    DOCKER_CMD=docker
else
     echo "ERROR: docker or nvidia-docker not found. Aborting." >&2
    exit 1
fi

DIRECTORY=`basename $1`

# ${DOCKER_CMD} run -ti --net=host --privileged -e DISPLAY=${DISPLAY} --rm --name ${DIRECTORY} ${DIRECTORY}

SCRIPT_DIR=$(cd ${DIRECTORY}; pwd)
${DOCKER_CMD} run -ti --net=host --privileged -e DISPLAY=${DISPLAY} -v ${SCRIPT_DIR}:/home/vizdoom/${DIRECTORY} --rm --name ${DIRECTORY} ${DIRECTORY}

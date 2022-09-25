#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Missing argument (path to target)"
fi

PACKAGE=tensorflow
VERSION=2.7.0
UNAME=`uname -m`
TARGET_DIR=/app/wheels/$PACKAGE/$VERSION/$UNAME

if compgen -G "$TARGET_DIR/*.whl" > /dev/null; then
    echo "Already has $PACKAGE $VERSION for $UNAME, skipping download"
else
    mkdir -p $TARGET_DIR

    if [ "$UNAME" == "aarch64" ]; then
        wget -P $TARGET_DIR/ https://cdn.edgeimpulse.com/build-system/wheels/aarch64/tensorflow-2.7.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        wget -P $TARGET_DIR/ https://cdn.edgeimpulse.com/build-system/wheels/aarch64/tensorflow_io-0.22.0-cp38-cp38-linux_aarch64.whl
        wget -P $TARGET_DIR/ https://cdn.edgeimpulse.com/build-system/wheels/aarch64/tensorflow_io_gcs_filesystem-0.22.0-cp38-cp38-linux_aarch64.whl
    else
        cd $TARGET_DIR
        pip3 download tensorflow==2.7.0 --no-deps
        pip3 download tensorflow-io==0.22.0 --no-deps
        pip3 download tensorflow-io-gcs-filesystem==0.22.0 --no-deps
    fi
fi

cd $TARGET_DIR
pip3 install *.whl

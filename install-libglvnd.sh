#!/bin/sh

set -e 
set -x

cd /opt
git clone --depth=1 https://github.com/NVIDIA/libglvnd
cd /opt/libglvnd
./autogen.sh
./configure
make -j4
make install

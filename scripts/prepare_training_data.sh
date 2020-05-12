#!/usr/bin/env bash

mkdir -p ./data/vibe_db
export PYTHONPATH="./:$PYTHONPATH"

# AMASS
python lib/data_utils/amass_utils.py --dir ./data/amass

# InstaVariety
# Comment this if you already downloaded the preprocessed file
python lib/data_utils/insta_utils.py --dir ./data/insta_variety

# 3DPW
python lib/data_utils/threedpw_utils.py --dir ./data/3dpw

# MPI-INF-3D-HP
python lib/data_utils/mpii3d_utils.py --dir ./data/mpi_inf_3dhp

# PoseTrack
python lib/data_utils/posetrack_utils.py --dir ./data/posetrack

# PennAction
python lib/data_utils/penn_action_utils.py --dir ./data/penn_action

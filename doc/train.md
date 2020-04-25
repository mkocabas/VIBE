# Training Instructions

Throughout the documentation we refer to VIBE root folder as `$ROOT`.

## Data Preparation
During training, VIBE uses precomputed image features to reduce training time. Thus, we process the datasets into a
standard format before using them for training. To obtain these standard training files, you need to run:

```shell script
source scripts/prepare_training_data.sh
```

This script will first create a folder for the dataset files `$ROOT/data/vibe_db`, then process each dataset and save
output files to this directory. Before proceeding, you need to download each of the datasets listed
below, then modify the `--dir` argument in the script to point the
directory of each dataset.



## Datasets

- **AMASS** (https://amass.is.tue.mpg.de)

Directory structure:

```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```

- **InstaVariety**

For your convenience, we uploaded the preprocessed InstaVariety data 
[here](https://owncloud.tuebingen.mpg.de/index.php/s/MKLnHtPjwn24y9C) (size: 18 GB). 
After downloading the file, put it under
`$ROOT/data/vibe_db`. Do not forget to verify checksum for sanity check: 
```
md5sum    : 8ec335d1d48bd54687ad5c9a6eeb2999
sha256sum : 7eadff77043cd85b49cbba8bfc9111c4305792ca64da1b92fb40fa702689dfa9
```

You may also preprocess the dataset yourself by downloading the 
[preprocessed tfrecords](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#pre-processed-tfrecords) 
provided by the authors of Temporal HMR.

Directory structure:
```shell script
insta_variety
|-- train
|   |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
|   |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
|   `-- ...
`-- test
    |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
    |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
    `-- ...
```

- **MPI-3D-HP** (http://gvv.mpi-inf.mpg.de/3dhp-dataset)

Donwload the dataset using the bash script provided by the authors. We will be using standard cameras only, so wall and ceiling
cameras aren't needed. Then, run this 
[script](https://gist.github.com/mkocabas/cc6fe78aac51f97859e45f46476882b6) to extract frames of videos.

Directory structure:
```shell script

mpi_inf_3dhp
|-- S1
|   |-- Seq1
|   |-- Seq2
|-- S2
|   |-- Seq1
|   |-- Seq2
|-- ...
`-- util
```

- **3DPW** (https://virtualhumans.mpi-inf.mpg.de/3DPW)

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```

- **PennAction** (http://dreamdragon.github.io/PennAction/)

Directory structure: 
```shell script
pennaction
|-- frames
|   |-- 0000
|   |-- 0001
|   |-- ...
`-- labels
    |-- 0000.mat
    |-- 0001.mat
    `-- ...
```

- **PoseTrack** (https://posetrack.net/)

Directory structure: 
```shell script
posetrack
|-- images
|   |-- train
|   |-- val
|   |-- test
`-- posetrack_data
    `-- annotations
        |-- train
        |-- val
        `-- test
```



## Training
Run the command below to start training.

```shell script
python train.py --cfg configs/config.yaml
```

See [`configs/config.yaml`](configs/config.yaml) or [`config.py`](lib/core/config.py) to 
play with different configurations.

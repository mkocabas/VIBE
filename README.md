# VIBE: Video Inference for Human Body Pose and Shape Estimation
[![report](https://img.shields.io/badge/arxiv-report-red)]() [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dFfwxZ52MN86FA6uFNypMEdFShd2euQA)

<p float="center">
  <img src="https://s5.gifyu.com/images/ezgif-6-fb456de304c9.gif" width="49%" />
  <img src="https://s5.gifyu.com/images/ezgif.com-optimize6d7c4d9d7251b20a.gif" width="49%" />
</p>

<sub>Video left: [https://www.youtube.com/watch?v=qlPRDVqYO74](https://www.youtube.com/watch?v=qlPRDVqYO74), Video right: [https://www.youtube.com/watch?v=Opry3F6aB1I](https://www.youtube.com/watch?v=Opry3F6aB1I)
</sub>

> [**VIBE: Video Inference for Human Body Pose and Shape Estimation**](https://arxiv.org/abs/1912.00000),            
> [Muhammed Kocabas](https://ps.is.tuebingen.mpg.de/person/mkocabas), [Nikos Athanasiou](https://ps.is.tuebingen.mpg.de/person/nathanasiou), 
[Michael J. Black](https://ps.is.tuebingen.mpg.de/person/black),        
> *ArXiv, 2019* 

## Features

_**V**ideo **I**nference for **B**ody Pose and Shape **E**stimation_ (VIBE) is a video pose and shape estimation method.
It predicts the parameters of SMPL body model for each frame of an input video. Pleaser refer [here](README.md#vibe-model) 
or the [arXiv report]() for further details.

This implementation:

- is the demo code for VIBE implemented purely in PyTorch,
- can work on arbitrary videos with multi person,
- supports both CPU and GPU inference (though GPU is way faster),
- is fast, up-to 30 FPS on a RTX2080Ti (see [this table](doc/demo.md#runtime-performance)),
- achieves SOTA results on 3DPW and MPI-INF-3DHP datasets,
- includes Temporal SMPLify implementation.

<p float="center">
  <img src="https://s5.gifyu.com/images/method_v2.gif" width="49%" />
  <img src="https://s5.gifyu.com/images/parkour.gif" width="49%" />
</p>


## Getting Started
VIBE has been implemented and tested on Ubuntu 18.04 with python >= 3.7. It supports both GPU and CPU inference.
If you don't have a suitable device, try running our Colab demo. 

Clone the repo:
```bash
git clone https://github.com/mkocabas/VIBE.git
```

Install the requirements using `pip` or `conda`:
```bash
# pip
bash install_pip.sh

# conda
bash install_conda.sh
```

## Running the Demo

We have prepared a nice demo code to run VIBE on arbitrary videos. 
First, you need download the required data(i.e our trained model and SMPL model parameters). To do this you can just run:

```bash
bash prepare_data.sh
```

Then, running the demo is as simple as this:

```bash
# Run on a local video
python demo.py --vid_file sample_video.mp4 --output_folder output/ --display

# Run on a YouTube video
python demo.py --vid_file https://www.youtube.com/watch?v=c4DAnQ6DtF8 --output_folder output/ --display
```

Refer to [`doc/demo.md`](doc/demo.md) for more details about the demo code.

## Google Colab
If you do not have a suitable environment to run this projects then you could give Google Colab a try. 
It allows you to run the project in the cloud, free of charge. You may try our Colab demo using the notebook we prepare: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dFfwxZ52MN86FA6uFNypMEdFShd2euQA)

## Evaluation

Here we compare VIBE with recent state-of-the-art methods on 3D pose estimation datasets. Evaluation metric is
Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) in mm.

| Models         | 3DPW &#8595; | MPI-INF-3DHP &#8595; | H36M &#8595; |
|----------------|:----:|:------------:|:----:|
| SPIN           | 59.2 |     67.5     | **41.1** |
| Temporal HMR   | 76.7 |     89.8     | 56.8 |
| VIBE           | 56.5 |     **63.4**     | 41.5 |
| VIBE + 3DPW    | **51.9** |     64.6     | 41.4 |

## Citation

```bibtex
@inproceedings{kocabas2019vibe,
  title={VIBE: Video Inference for Human Body Pose and Shape Estimation},
  author={Kocabas, Muhammed and Athanasiou, Nikos and Black, Michael J.},
  journal={arXiv preprint arXiv:1912.00000},
  year={2019}
}
```

## License
This code is freely available for **non-commercial scientific research purposes**, and may be redistributed under these conditions. Please, see the [LICENSE](LICENSE) for details. Third-party datasets and softwares are subject to their respective licenses.


## References
We indicate if a function or script is borrowed externally inside each file. Here are some great resources we 
benefit:

- Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [Temporal HMR](https://github.com/akanazawa/human_dynamics).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions are borrowed from [Kornia](https://github.com/kornia/kornia).
- Pose tracker is from [STAF](https://github.com/soulslicer/openpose/tree/staf).


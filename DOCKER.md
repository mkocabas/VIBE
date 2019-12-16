# Docker VIBE

## Runing the CPU image


## Running the GPU image

This requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) in order for the local GPUs to be made accessible by the container.

The following steps are required:

- install `nvidia-docker`: https://github.com/NVIDIA/nvidia-docker
- run with
```bash
nvidia-docker run mkocabas/vibe:ubuntu-gpu
```

Notes:
- `nvidia-docker` requires docker >= 1.9

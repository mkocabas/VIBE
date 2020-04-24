# Evaluation

Run the commands below to evaluate a pretrained model.

```shell script
python eval.py --cfg configs/config.yaml
```

Change the `TRAIN.PRETRAINED` field of the config file to the checkpoint you would like to evaluate.
You should be able to obtain the output below:

```shell script
# TRAIN.PRETRAINED = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
...Evaluating on 3DPW test set...
MPJPE: 93.5881, PA-MPJPE: 56.5608, PVE: 113.4118, ACCEL: 27.1242, ACCEL_ERR: 27.9877

# TRAIN.PRETRAINED = 'data/vibe_data/vibe_model_w_3dpw.pth.tar'
...Evaluating on 3DPW test set...
MPJPE: 82.9725, PA-MPJPE: 52.0008, PVE: 99.1107, ACCEL: 22.3731, ACCEL_ERR: 23.4265
```

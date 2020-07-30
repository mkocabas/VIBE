# Demo

## Flags

- `--vid_file (str)`: Path to input video file or a YouTube link. If you provide a YouTube link it will be downloaded
to a temporary folder and then processed.

- `--output_folder (str)`: Path to folder to store the VIBE predictions and output renderings.

- `--tracking_method (str), default=bbox`: Defines the tracking method to compute bboxes and tracklets of people in the input video.
 Available options are `bbox` or `pose`. `bbox` tracking is available [here](https://github.com/mkocabas/multi-person-tracker) 
 as a standalone python package. For `pose` tracking, you need to install 
 [STAF](https://github.com/soulslicer/openpose/tree/staf), extension of OpenPose to 
 multi-person posetracking recently introduced in [1]().
 
- `--detector (str), default=yolo`: Defines the type of detector to be used by `bbox` tracking method if enabled. Available options are
`maskrcnn` and `yolo`. `maskrcnn` is more accurate but slower compared to `yolo`. Refer to [speed comparison](demo.md#runtime-performance) for further information.

- `--yolo_img_size (int), default=416`: Input image size of YOLO detector.

- `--tracker_batch_size (int), default=12`: Batch size of the bbox tracker. If you get memory error, you need to reduce it.  

- `--staf_dir (str)`: Path to folder where STAF pose tracker installed. This path should point to the main directory of staf.

- `--vibe_batch_size (int), default=450`: Batch size of VIBE model.

- `--display`: Enable this flag if you want to visualize the output of tracking and pose & shape estimation interactively.

- `--run_smplify`: Enable this flag if you want to refine the results of VIBE using Temporal SMPLify algorithm.
For this option, you have to set `--tracking_method` option to `pose`.

- `--no_render`: This flag disables the final rendering of VIBE results. Useful if you only want to get VIBE predictions.

- `--wireframe`: Enable this if you would like to render wireframe meshes in the final rendering. 

- `--sideview`: Render the output meshes from an alternate viewpoint. Default alternate viewpoint is -90 degrees in y axis.
Note that this option doubles the rendering time.

- `--save_obj`: Save output meshes as .obj files.

## Examples
- Run VIBE on a video file using bbox tracker and visualize the results with wireframe meshes:
```bash
python demo_video.py --vid_file sample_video.mp4 --output_folder output/ --tracking_method bbox --detector maskrcnn --display --wireframe
```

- Run VIBE on a YouTube video using pose tracker and run Temporal SMPLify to further refine the predictions:
```bash
python demo_video.py --vid_file sample_video.mp4 --output_folder output/ --tracking_method pose --display --run_smplify
```

- Change the default batch sizes to avoid possible memory errors:
```bash
python demo_video.py --vid_file sample_video.mp4 --output_folder output/ --tracker_batch_size 2 --vibe_batch_size 64
```

## Output Format

If demo finishes succesfully, it needs to create a file named `vibe_output.pkl` in the `--output_folder`.
We can inspect what this file contains by:

```python
>>> import joblib # you may use native pickle here as well

>>> output = joblib.load('output/group_dance/vibe_output.pkl') 

>>> print(output.keys())  
                                                                                                                                                                                                                                                                                                                                                                                              
dict_keys([1, 2, 3, 4]) # these are the track ids for each subject appearing in the video

>>> for k,v in output[1].items(): print(k,v.shape) 

pred_cam (n_frames, 3)      # weak perspective camera parameters in cropped image space (s,tx,ty)
orig_cam (n_frames, 4)      # weak perspective camera parameters in original image space (sx,sy,tx,ty)
verts (n_frames, 6890, 3)   # SMPL mesh vertices
pose (n_frames, 72)         # SMPL pose parameters
betas (n_frames, 10)        # SMPL body shape parameters
joints3d (n_frames, 49, 3)  # SMPL 3D joints
joints2d (n_frames, 21, 3)  # 2D keypoint detections by STAF if pose tracking enabled otherwise None
bboxes (n_frames, 4)        # bbox detections (cx,cy,w,h)
frame_ids (n_frames,)       # frame ids in which subject with tracking id #1 appears

```
You can find the names & order of 3d joints [here](https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/kp_utils.py#L212) and 2D joints [here](https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/kp_utils.py#L187).

## Runtime Performance
Here is the breakdown of runtime speeds per step namely tracking and VIBE. This results are obtained by running VIBE
on a [video](https://www.youtube.com/watch?v=Opry3F6aB1I) containing 5 people.

```bash
python demo.py --vid_file https://www.youtube.com/watch?v=Opry3F6aB1I --output_folder output/ --vibe_batch_size 32 --no_render
```

| Tracker         |    GPU    | Tracking Time (ms/img) | Tracking FPS | VIBE Time (ms/image) | VIBE FPS | Total FPS |
|-----------------|:---------:|:----------------------:|:------------:|:--------------------:|:--------:|:---------:|
| STAF-pose       | RTX2080Ti |          23.2          |      43      |         16.1         |    61    |     21    |
| MaskRCNN-bbox   | RTX2080Ti |          68.0          |      15      |         16.1         |    61    |     11    |
| YOLOv3-416-bbox | RTX2080Ti |          12.7          |      79      |         16.1         |    61    |     29    |
| YOLOv3-608-bbox | RTX2080Ti |          22.2          |      45      |         16.1         |    61    |     23    |

**Note**: Above table does not include the time spent during rendering of the final output. 
We use pyrender with GPU accelaration and it takes 2-3 FPS per image. Please let us know if you know any faster alternative.

## References
[1] Pose tracker is from [STAF implementation](https://github.com/soulslicer/openpose/tree/staf)

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.vis import batch_visualize_preds
from lib.data_utils.img_utils import get_single_image_crop, convert_cvimg_to_tensor


def extract_features(model, video, bbox, debug=False, batch_size=200, kp_2d=None, dataset=None, scale=1.3):
    '''
    :param model: pretrained HMR model, use lib/models/hmr.py:get_pretrained_hmr()
    :param video: video filename, torch.Tensor in shape (num_frames,W,H,C)
    :param bbox: bbox array in shape (T,4)
    :param debug: boolean, true if you want to debug HMR predictions
    :param batch_size: batch size for HMR input
    :return: features: resnet50 features np.ndarray -> shape (num_frames, 4)
    '''
    device = 'cuda'

    if isinstance(video, torch.Tensor) or isinstance(video, np.ndarray):
        video = video
    elif isinstance(video, str):
        if os.path.isfile(video):
            video, _, _ = torchvision.io.read_video(video)
        else:
            raise ValueError(f'{video} is not a valid file.')
    else:
        raise ValueError(f'Unknown type {type(video)} for video object')

    # For debugging ground truth 2d keypoints
    if debug and kp_2d is not None:
        import cv2
        if isinstance(video[0], np.str_):
            print(video[0])
            frame = cv2.cvtColor(cv2.imread(video[0]), cv2.COLOR_BGR2RGB)
        elif isinstance(video[0], np.ndarray):
            frame = video[0]
        else:
            frame = video[0].numpy()
        for i in range(kp_2d.shape[1]):
            frame = cv2.circle(
                frame.copy(),
                (int(kp_2d[0,i,0]), int(kp_2d[0,i,1])),
                thickness=3,
                color=(255,0,0),
                radius=3,
            )

        plt.imshow(frame)
        plt.show()

    if dataset == 'insta':
        video = torch.cat(
            [convert_cvimg_to_tensor(image).unsqueeze(0) for image in video], dim=0
        ).to(device)
    else:
        # crop bbox locations
        video = torch.cat(
            [get_single_image_crop(image, bbox, scale=scale).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
        ).to(device)

    features = []

    # split video into batches of frames
    frames = torch.split(video, batch_size)

    with torch.no_grad():
        for images in frames:

            if not debug:
                pred = model.feature_extractor(images)
                features.append(pred.cpu())
                del pred, images
            else:
                preds = model(images)
                dataset = 'spin' # dataset if dataset else 'common'
                result_image = batch_visualize_preds(
                    images,
                    preds[-1],
                    target_exists=False,
                    max_images=4,
                    dataset=dataset,
                )

                plt.figure(figsize=(19.2, 10.8))
                plt.axis('off')
                plt.imshow(result_image)
                plt.show()

                del preds, images
                return 0

        features = torch.cat(features, dim=0)

    return features.numpy()

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

import torch
import numpy as np

from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.one_euro_filter import OneEuroFilter


def smooth_pose(pred_pose, pred_betas, min_cutoff=0.004, beta=0.7):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    smpl = SMPL(model_path=SMPL_MODEL_DIR)

    pred_pose_hat = np.zeros_like(pred_pose)

    # initialize
    pred_pose_hat[0] = pred_pose[0]

    pred_verts_hat = []
    pred_joints3d_hat = []

    smpl_output = smpl(
        betas=torch.from_numpy(pred_betas[0]).unsqueeze(0),
        body_pose=torch.from_numpy(pred_pose[0, 1:]).unsqueeze(0),
        global_orient=torch.from_numpy(pred_pose[0, 0:1]).unsqueeze(0),
    )
    pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
    pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1

        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose

        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[idx]).unsqueeze(0),
            body_pose=torch.from_numpy(pred_pose_hat[idx, 1:]).unsqueeze(0),
            global_orient=torch.from_numpy(pred_pose_hat[idx, 0:1]).unsqueeze(0),
        )
        pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    return np.vstack(pred_verts_hat), pred_pose_hat, np.vstack(pred_joints3d_hat)
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

import sys
sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.models import spin
from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils.feature_extractor import extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set, debug=False):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        'valid': [],
    }

    model = spin.get_pretrained_hmr()

    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]

    J_regressor = None

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, seq in tqdm(enumerate(sequences)):

        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')

        data = pkl.load(open(data_file, 'rb'), encoding='latin1')

        img_dir = osp.join(folder, 'imageFiles', seq)

        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)

        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            trans = torch.from_numpy(data['trans'][p_id]).float()
            j2d = data['poses2d'][p_id].transpose(0,2,1)
            cam_pose = data['cam_poses']
            campose_valid = data['campose_valid'][p_id]

            # ======== Align the mesh params ======== #
            rot = pose[:, :3]
            rot_mat = batch_rodrigues(rot)

            Rc = torch.from_numpy(cam_pose[:, :3, :3]).float()
            Rs = torch.bmm(Rc, rot_mat.reshape(-1, 3, 3))
            rot = rotation_matrix_to_angle_axis(Rs)
            pose[:, :3] = rot
            # ======== Align the mesh params ======== #

            output = smpl(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
            # verts = output.vertices
            j3d = output.joints

            if J_regressor is not None:
                vertices = output.vertices
                J_regressor_batch = J_regressor[None, :].expand(vertices.shape[0], -1, -1).to(vertices.device)
                j3d = torch.matmul(J_regressor_batch, vertices)
                j3d = j3d[:, H36M_TO_J14, :]

            img_paths = []
            for i_frame in range(num_frames):
                img_path = os.path.join(img_dir + '/image_{:05d}.jpg'.format(i_frame))
                img_paths.append(img_path)

            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

            # process bbox_params
            c_x = bbox_params[:,0]
            c_y = bbox_params[:,1]
            scale = bbox_params[:,2]
            w = h = 150. / scale
            w = h = h * 1.1
            bbox = np.vstack([c_x,c_y,w,h]).T

            # process keypoints
            j2d[:, :, 2] = j2d[:, :, 2] > 0.3  # set the visibility flags
            # Convert to common 2d keypoint format
            perm_idxs = get_perm_idxs('3dpw', 'common')
            perm_idxs += [0, 0]  # no neck, top head
            j2d = j2d[:, perm_idxs]
            j2d[:, 12:, 2] = 0.0

            # print('j2d', j2d[time_pt1:time_pt2].shape)
            # print('campose', campose_valid[time_pt1:time_pt2].shape)

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2]
            dataset['vid_name'].append(np.array([f'{seq}_{p_id}']*num_frames)[time_pt1:time_pt2])
            dataset['frame_id'].append(np.arange(0, num_frames)[time_pt1:time_pt2])
            dataset['img_name'].append(img_paths_array)
            dataset['joints3D'].append(j3d.numpy()[time_pt1:time_pt2])
            dataset['joints2D'].append(j2d[time_pt1:time_pt2])
            dataset['shape'].append(shape.numpy()[time_pt1:time_pt2])
            dataset['pose'].append(pose.numpy()[time_pt1:time_pt2])
            dataset['bbox'].append(bbox)
            dataset['valid'].append(campose_valid[time_pt1:time_pt2])

            features = extract_features(model, img_paths_array, bbox,
                                        kp_2d=j2d[time_pt1:time_pt2], debug=debug, dataset='3dpw', scale=1.2)
            dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    # Filter out keypoints
    indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][indices_to_use]

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/3dpw')
    args = parser.parse_args()

    debug = False

    dataset = read_data(args.dir, 'validation', debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, '3dpw_val_db.pt'))

    dataset = read_data(args.dir, 'test', debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, '3dpw_test_db.pt'))

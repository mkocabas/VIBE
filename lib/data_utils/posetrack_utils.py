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

import glob
import joblib
import argparse
import numpy as np
import json
import os.path as osp

from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.feature_extractor import extract_features
from lib.data_utils.kp_utils import get_posetrack_original_kp_names, convert_kps

def read_data(folder, set):
    dataset = {
        'img_name' : [] ,
        'joints2D': [],
        'bbox': [],
        'vid_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    file_names = glob.glob(osp.join(folder, 'posetrack_data/annotations/', f'{set}/*.json'))
    file_names = sorted(file_names)
    nn_corrupted = 0
    tot_frames = 0
    min_frame_number = 8

    for fid,fname in tqdm_enumerate(file_names):
        if fname == osp.join(folder, 'annotations/train/021133_mpii_train.json'):
            continue

        with open(fname, 'r') as entry:
            anns = json.load(entry)
        # num_frames = anns['images'][0]['nframes']
        anns['images'] = [item for item in anns['images'] if item['is_labeled'] ]
        num_frames = len(anns['images'])
        frame2imgname = dict()
        for el in anns['images']:
            frame2imgname[el['frame_id']] = el['file_name']

        num_people = -1
        for x in anns['annotations']:
            if num_people < x['track_id']:
                num_people = x['track_id']
        num_people += 1
        posetrack_joints = get_posetrack_original_kp_names()
        idxs = [anns['categories'][0]['keypoints'].index(h) for h in posetrack_joints if h in anns['categories'][0]['keypoints']]
        for x in anns['annotations']:
            kps = np.array(x['keypoints']).reshape((17,3))
            kps = kps[idxs,:]
            x['keypoints'] = list(kps.flatten())

        tot_frames += num_people * num_frames
        for p_id in range(num_people):

            annot_pid = [(item['keypoints'], item['bbox'], item['image_id'])
                         for item in anns['annotations']
                         if item['track_id'] == p_id and not(np.count_nonzero(item['keypoints']) == 0)  ]

            if len(annot_pid) < min_frame_number:
                nn_corrupted += len(annot_pid)
                continue

            bbox = np.zeros((len(annot_pid),4))
            # perm_idxs = get_perm_idxs('posetrack', 'common')
            kp_2d = np.zeros((len(annot_pid), len(annot_pid[0][0])//3 ,3))
            img_paths = np.zeros((len(annot_pid)))

            for i, (key2djnts, bbox_p, image_id) in enumerate(annot_pid):

                if (bbox_p[2]==0 or bbox_p[3]==0) :
                    nn_corrupted +=1
                    continue

                img_paths[i] = image_id
                key2djnts[2::3] = len(key2djnts[2::3])*[1]

                kp_2d[i,:] = np.array(key2djnts).reshape(int(len(key2djnts)/3),3) # [perm_idxs, :]
                for kp_loc in kp_2d[i,:]:
                    if kp_loc[0] == 0 and kp_loc[1] == 0:
                        kp_loc[2] = 0


                x_tl = bbox_p[0]
                y_tl = bbox_p[1]
                w = bbox_p[2]
                h = bbox_p[3]
                bbox_p[0] = x_tl + w / 2
                bbox_p[1] = y_tl + h / 2
                #

                w = h = np.where(w / h > 1, w, h)
                w = h = h * 0.8
                bbox_p[2] = w
                bbox_p[3] = h
                bbox[i, :] = bbox_p

            img_paths = list(img_paths)
            img_paths = [osp.join(folder, frame2imgname[item]) if item != 0 else 0 for item in img_paths ]

            bbx_idxs = []
            for bbx_id, bbx in enumerate(bbox):
                if np.count_nonzero(bbx) == 0:
                    bbx_idxs += [bbx_id]

            kp_2d = np.delete(kp_2d, bbx_idxs, 0)
            img_paths = np.delete(np.array(img_paths), bbx_idxs, 0)
            bbox = np.delete(bbox, np.where(~bbox.any(axis=1))[0], axis=0)

            # Convert to common 2d keypoint format
            if bbox.size == 0 or bbox.shape[0] < min_frame_number:
                nn_corrupted += 1
                continue

            kp_2d = convert_kps(kp_2d, src='posetrack', dst='spin')

            dataset['vid_name'].append(np.array([f'{fname}_{p_id}']*img_paths.shape[0]))
            dataset['img_name'].append(np.array(img_paths))
            dataset['joints2D'].append(kp_2d)
            dataset['bbox'].append(np.array(bbox))

            # compute_features
            features = extract_features(
                model,
                np.array(img_paths),
                bbox,
                kp_2d=kp_2d,
                dataset='spin',
                debug=False,
            )

            assert kp_2d.shape[0] == img_paths.shape[0] == bbox.shape[0]

            dataset['features'].append(features)


    print(nn_corrupted, tot_frames)
    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    for k,v in dataset.items():
        print(k, v.shape)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/posetrack')
    args = parser.parse_args()

    dataset_train = read_data(args.dir, 'train')
    joblib.dump(dataset_train, osp.join(VIBE_DB_DIR, 'posetrack_train_db.pt'))

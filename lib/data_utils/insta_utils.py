# Some functions are borrowed from
# https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/insta_variety_to_tfrecords.py
# Adhere to their licence to use these functions

"""
Visualizes tfrecords.
Sample usage:
python -m src.datasets.visualize_tfrecords --data_rootdir /scratch3/kanazawa/hmmr_tfrecords_release_test/ --dataset penn_action --split test
"""
import os
import sys
import h5py
sys.path.append('.')

import argparse
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from lib.models import spin
from lib.utils.vis import draw_skeleton
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.feature_extractor import extract_features

class ImageCoder(object):
    """
    Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/
        build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def read_from_example(serialized_ex):
    """
    Returns data from an entry in test tfrecord.

    Args:
        serialized_ex (str).

    Returns:
        dict. Keys:
            N (1).
            centers (Nx2).
            kps (Nx19x3).
            gt3ds (Nx14x3).
            images (Nx224x224x3).
            im_shapes (Nx2).
            im_paths (N).
            poses (Nx24x3).
            scales (N).
            shape (10).
            start_pts (Nx2).
            time_pts (2).
    """
    coder = ImageCoder()
    example = tf.train.Example()
    example.ParseFromString(serialized_ex)
    features = example.features.feature

    # Load features from example.
    N = features['meta/N'].int64_list.value[0]
    im_datas = features['image/encoded'].bytes_list.value
    centers = features['image/centers'].int64_list.value
    xys = features['image/xys'].float_list.value
    face_pts = features['image/face_pts'].float_list.value
    toe_pts = features['image/toe_pts'].float_list.value
    vis = features['image/visibilities'].int64_list.value
    scales = np.array(features['image/scale_factors'].float_list.value)
    gt3ds = features['mosh/gt3ds'].float_list.value
    poses = features['mosh/poses'].float_list.value
    shape = features['mosh/shape'].float_list.value
    time_pts = features['meta/time_pts'].int64_list.value
    start_pts = np.array(features['image/crop_pts'].int64_list.value)
    im_shapes = features['image/heightwidths'].int64_list.value
    im_paths = features['image/filenames'].bytes_list.value

    # Process and reshape features.
    images = [coder.decode_jpeg(im_data) for im_data in im_datas]
    centers = np.array(centers).reshape((N, 2))
    gt3ds = np.array(gt3ds).reshape((N, -1, 3))
    gt3ds = gt3ds[:, :14]  # Don't want toes_pts or face_pts
    xys = np.array(xys).reshape((N, 2, 14))
    vis = np.array(vis, dtype=np.float).reshape((N, 1, 14))
    face_pts = np.array(face_pts).reshape((N, 3, 5))
    toe_pts = np.array(toe_pts).reshape((N, 3, 6))
    kps = np.dstack((
        np.hstack((xys, vis)),
        face_pts,
        toe_pts,
    ))
    kps = np.transpose(kps, axes=[0, 2, 1])
    poses = np.array(poses).reshape((N, 24, 3))
    shape = np.array(shape)
    start_pts = np.array(start_pts).reshape((N, 2))
    im_shapes = np.array(im_shapes).reshape((N, 2))

    return {
        'N': N,
        'centers': centers,
        'kps': kps,
        'gt3ds': gt3ds,
        'images': images,
        'im_shapes': im_shapes,
        'im_paths': im_paths,
        'poses': poses,
        'scales': scales,
        'shape': shape,
        'start_pts': start_pts,
        'time_pts': time_pts,
    }


def visualize_tfrecords(fpaths):
    sess = tf.Session()
    for fname in fpaths:
        print(fname)
        for serialized_ex in tf.python_io.tf_record_iterator(fname):
            example = tf.train.Example()
            example.ParseFromString(serialized_ex)
            # import ipdb; ipdb.set_trace()
            # Now these are sequences.
            N = int(example.features.feature['meta/N'].int64_list.value[0])
            print(N)
            # This is a list of length N
            images_data = example.features.feature[
                'image/encoded'].bytes_list.value

            xys = example.features.feature['image/xys'].float_list.value
            xys = np.array(xys).reshape(-1, 2, 14)

            face_pts = example.features.feature[
                'image/face_pts'].float_list.value
            face_pts = np.array(face_pts).reshape(-1, 3, 5)

            toe_pts = example.features.feature[
                'image/toe_pts'].float_list.value

            if len(toe_pts) == 0:
                toe_pts = np.zeros(xys.shape[0], 3, 6)

            toe_pts = np.array(toe_pts).reshape(-1, 3, 6)

            visibles = example.features.feature[
                'image/visibilities'].int64_list.value
            visibles = np.array(visibles).reshape(-1, 1, 14)
            centers = example.features.feature[
                'image/centers'].int64_list.value
            centers = np.array(centers).reshape(-1, 2)

            if 'image/phis' in example.features.feature.keys():
                phis = example.features.feature['image/phis'].float_list.value
                phis = np.array(phis)

            for i in range(N):
                image = sess.run(tf.image.decode_jpeg(images_data[i], channels=3))
                kp = np.vstack((xys[i], visibles[i]))
                faces = face_pts[i]

                toes = toe_pts[i]
                kp = np.hstack((kp, faces, toes))
                if 'image/phis' in example.features.feature.keys():
                    # Preprocessed, so kps are in [-1, 1]
                    img_shape = image.shape[0]
                    vis = kp[2, :]
                    kp = ((kp[:2, :] + 1) * 0.5) * img_shape
                    kp = np.vstack((kp, vis))

                plt.ion()
                plt.clf()
                plt.figure(1)

                skel_img = draw_skeleton(image, kp.T, dataset='insta', unnormalize=False)
                plt.imshow(skel_img)
                plt.title(f'{i}')

                plt.axis('off')
                plt.pause(0.5)


def read_single_record(fname):
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints2D': [], # should contain openpose keypoints only
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    sess = tf.Session()

    for vid_idx, serialized_ex in tqdm(enumerate(tf.python_io.tf_record_iterator(fname))):
        example = tf.train.Example()
        example.ParseFromString(serialized_ex)

        N = int(example.features.feature['meta/N'].int64_list.value[0])

        # print(fname, vid_idx, N)
        # This is a list of length N
        images_data = example.features.feature[
            'image/encoded'].bytes_list.value

        xys = example.features.feature['image/xys'].float_list.value
        xys = np.array(xys).reshape(-1, 2, 14)

        face_pts = example.features.feature[
            'image/face_pts'].float_list.value
        face_pts = np.array(face_pts).reshape(-1, 3, 5)

        toe_pts = example.features.feature[
            'image/toe_pts'].float_list.value

        if len(toe_pts) == 0:
            toe_pts = np.zeros(xys.shape[0], 3, 6)

        toe_pts = np.array(toe_pts).reshape(-1, 3, 6)

        visibles = example.features.feature[
            'image/visibilities'].int64_list.value
        visibles = np.array(visibles).reshape(-1, 1, 14)

        video = []
        kp_2d = []
        for i in range(N):
            image = np.expand_dims(sess.run(tf.image.decode_jpeg(images_data[i], channels=3)), axis=0)
            video.append(image)

            kp = np.vstack((xys[i], visibles[i]))
            faces = face_pts[i]
            toes = toe_pts[i]

            kp = np.hstack((kp, faces, toes))

            if 'image/phis' in example.features.feature.keys():
                # Preprocessed, so kps are in [-1, 1]
                img_shape = 224  # image.shape[0]
                vis = kp[2, :]
                kp = ((kp[:2, :] + 1) * 0.5) * img_shape
                kp = np.vstack((kp, vis))

            kp_2d.append(np.expand_dims(kp.T, axis=0))

        video = np.concatenate(video, axis=0)
        kp_2d = np.concatenate(kp_2d, axis=0)

        vid_name = f'{fname}-{vid_idx}'
        frame_id = np.arange(N)
        joints2D = kp_2d

        dataset['vid_name'].append(np.array([vid_name] * N))
        dataset['frame_id'].append(frame_id)
        dataset['joints2D'].append(joints2D)

        features = extract_features(model, video, bbox=None, kp_2d=kp_2d, dataset='insta', debug=False)
        dataset['features'].append(features)

        print(features.shape)
        assert features.shape[0] == N

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    for k,v in dataset.items():
        print(k, len(v))

    return dataset


def save_hdf5(filename, db):
    with h5py.File(filename, 'w') as f:
        for k, v in db.items():
            if k == 'vid_name':
                v = np.array(v, dtype=np.string_)
            f.create_dataset(k, data=v)


def concatenate_annotations():
    ds = {
        'vid_name': [],
        'frame_id': [],
        'joints2D': [],
        'features': [],
    }

    for i in range(273):
        filename = osp.join(VIBE_DB_DIR, 'insta_parts', f'insta_train_part_{i}.h5')
        print(filename)
        with h5py.File(filename, 'r') as f:
            for k in ds.keys():
                ds[k].append(f[k].value)

    for k in ds.keys():
        ds[k] = np.concatenate(ds[k])

    print('Saving Insta Variety dataset!..')
    db_file = osp.join(VIBE_DB_DIR, 'insta_train_db.h5')
    save_hdf5(db_file, ds)
    print('Saved Insta Variety dataset!...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/insta_variety')
    args = parser.parse_args()

    split = 'train'
    fpaths = glob(f'{args.dir}/{split}/*.tfrecord')
    fpaths = sorted(fpaths)

    os.makedirs(osp.join(VIBE_DB_DIR, 'insta_parts'), exist_ok=True)

    for idx, fp in enumerate(fpaths):
        dataset = read_single_record(fp)

        db_file = osp.join(VIBE_DB_DIR, 'insta_parts', f'insta_train_part_{idx}.h5')

        save_hdf5(db_file, dataset)

    concatenate_annotations()






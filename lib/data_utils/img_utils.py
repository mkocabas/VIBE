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
import cv2
import torch

import random
import numpy as np
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows

def get_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def do_augmentation(scale_factor=0.3, color_factor=0.2):
    scale = random.uniform(1.2, 1.2+scale_factor)
    # scale = np.clip(np.random.randn(), 0.0, 1.0) * scale_factor + 1.2
    rot = 0 # np.clip(np.random.randn(), -2.0, 2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    do_flip = False # aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans

def crop_image(image, kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    # get augmentation params
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.3, 0, False, [1.0, 1.0, 1.0]

    # generate image patch
    image, trans = generate_patch_image_cv(
        image,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return image, kp_2d, trans

def transfrom_keypoints(kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.2, 0, False, [1.0, 1.0, 1.0]

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans

def get_image_crops(image_file, bboxes):
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    crop_images = []
    for bb in bboxes:
        c_y, c_x = (bb[0]+bb[2]) // 2, (bb[1]+bb[3]) // 2
        h, w = bb[2]-bb[0], bb[3]-bb[1]
        w = h = np.where(w / h > 1, w, h)
        crop_image, _ = generate_patch_image_cv(
            cvimg=image.copy(),
            c_x=c_x,
            c_y=c_y,
            bb_width=w,
            bb_height=h,
            patch_width=224,
            patch_height=224,
            do_flip=False,
            scale=1.3,
            rot=0,
        )
        crop_image = convert_cvimg_to_tensor(crop_image)
        crop_images.append(crop_image)

    batch_image = torch.cat([x.unsqueeze(0) for x in crop_images])
    return batch_image

def get_single_image_crop(image, bbox, scale=1.3):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, _ = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=224,
        patch_height=224,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image

def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    if kp_2d is not None:
        for n_jt in range(kp_2d.shape[0]):
            kp_2d[n_jt, :2] = trans_point2d(kp_2d[n_jt], trans)

    raw_image = crop_image.copy()

    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image, raw_image, kp_2d

def read_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    return convert_cvimg_to_tensor(image)

def convert_cvimg_to_tensor(image):
    transform = get_default_transform()
    image = transform(image)
    return image

def torch2numpy(image):
    image = image.detach().cpu()
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    image = inv_normalize(image)
    image = image.clamp(0., 1.)
    image = image.numpy() * 255.
    image = np.transpose(image, (1, 2, 0))
    return image.astype(np.uint8)

def torch_vid2numpy(video):
    video = video.detach().cpu().numpy()
    # video = np.transpose(video, (0, 2, 1, 3, 4)) # NCTHW->NTCHW
    # Denormalize
    mean = np.array([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255])
    std = np.array([1 / 0.229, 1 / 0.224, 1 / 0.255])

    mean = mean[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    std = std[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]

    video = (video - mean) / std # [:, :, i, :, :].sub_(mean[i]).div_(std[i]).clamp_(0., 1.).mul_(255.)
    video = video.clip(0.,1.) * 255
    video = video.astype(np.uint8)
    return video

def get_bbox_from_kp2d(kp_2d):
    # get bbox
    if len(kp_2d.shape) > 2:
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox

def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d

def get_default_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform

def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices
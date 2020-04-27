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

import cv2
import math
import time
import torch
import trimesh
import pyrender
import numpy as np
from matplotlib import pyplot as plt

from lib.data_utils import kp_utils
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, get_smpl_faces
from lib.data_utils.img_utils import torch2numpy, torch_vid2numpy, normalize_2d_kp


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        P[0, 3] = self.translation[0] * self.scale
        P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1
        return P


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors


def render_image(img, verts, cam, faces=None, angle=None, axis=None, resolution=224, output_fn=None):
    if faces is None:
        faces = get_smpl_faces()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
    mesh.apply_transform(Rx)

    if angle and axis:
        R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
        mesh.apply_transform(R)

    if output_fn:
        mesh.export(output_fn)
        camera_translation = np.array([-cam[1], cam[2], 2 * 5000. / (img.shape[0] * cam[0] + 1e-9)])
        np.save(output_fn.replace('.obj', '.npy'), camera_translation)

        # Save the rotated mesh
        # R = trimesh.transformations.rotation_matrix(math.radians(270), [0,1,0])
        # rotated_mesh = mesh.copy()
        # rotated_mesh.apply_transform(R)
        # rotated_mesh.export(output_fn.replace('.obj', '_rot.obj'))



    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3)
                           )

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)

    camera = WeakPerspectiveCamera(
        scale=cam[0],
        translation=cam[1:],
        zfar=1000.
    )
    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = [0, 1, 1]
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = [1, 1, 2]
    scene.add(light, pose=light_pose)


    r = pyrender.OffscreenRenderer(viewport_width=resolution,
                                   viewport_height=resolution,
                                   point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # color = color[:, ::-1, :]
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

    output_img = color[:, :, :-1] * valid_mask + (1 - valid_mask) * img

    image = output_img.astype(np.uint8)
    text = f's: {cam[0]:.2f}, tx: {cam[1]:.2f}, ty: {cam[2]:.2f}'
    cv2.putText(image, text, (5, 10), 0, 0.4, color=(0,255,0))

    return image


def draw_SMPL_joints2D(joints2D, image, kintree_table=None, color='red'):
    rcolor = get_colors()['red'].tolist()
    lcolor = get_colors()['blue'].tolist()
    # color = get_colors()[color].tolist()
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]

        color = lcolor if i % 2 == 0 else rcolor

        pt1, pt2 = (joints2D[j1, 0], joints2D[j1, 1]), (joints2D[j2, 0], joints2D[j2, 1])
        cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=2)

        cv2.circle(image, pt1, 4, color, -1)
        cv2.circle(image, pt2, 4, color, -1)

    # for i in range(joints2D.shape[0]):
    #     color = lcolor if i % 2 == 0 else rcolor
    #     pt1 = (joints2D[i, 0], joints2D[i, 1])
    #     cv2.circle(image, pt1, 4, color, -1)

    return image


def show3Dpose(channels, ax, radius=40, lcolor='#ff0000', rcolor='#0000ff'):
    vals = channels

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)

    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def visualize_sequence(sequence):

    seqlen, size = sequence.shape
    sequence = sequence.reshape((seqlen, -1, 3))


    fig = plt.figure(figsize=(12, 7))

    for i in range(seqlen):
        ax = fig.add_subplot('111', projection='3d', aspect=1)
        show3Dpose(sequence[i], ax, radius=0.6)
        ax.view_init(-75, -90)

        plt.draw()
        plt.pause(0.01)
        plt.cla()

    plt.close()

def visualize_preds(image, preds, target=None, target_exists=True, dataset='common', vis_hmr=False):
    with torch.no_grad():
        if isinstance(image, torch.Tensor):
            image = torch2numpy(image)
            # import random
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f'sample_images/{random.randint(0,100)}.jpg', image)

    pred_theta  = preds['theta']
    pred_camera = pred_theta[:3]
    # pred_pose   = pred_theta[3:75]
    # pred_shape  = pred_theta[75:]
    pred_kp_2d  = preds['kp_2d']
    pred_verts = preds['verts']

    if target_exists:
        target_kp_2d  = target['kp_2d']

    pred_kp_2d = np.concatenate([pred_kp_2d, np.ones((pred_kp_2d.shape[0], 1))], axis=-1)

    faces = get_smpl_faces()

    pred_image = draw_skeleton(image.copy(), pred_kp_2d, dataset=dataset)
    if target_exists:
        if vis_hmr:
            target_verts = target['verts']
            target_cam = target['cam']

            target_image = render_image(
                img=image.copy(),
                verts=target_verts,
                faces=faces,
                cam=target_cam
            )
        else:
            target_image = draw_skeleton(image.copy(), target_kp_2d, dataset=dataset)


    render = render_image(
        img=image.copy(),
        verts=pred_verts,
        faces=faces,
        cam=pred_camera
    )

    white_img = np.zeros_like(image)
    render_side = render_image(
        img=white_img.copy(),
        verts=pred_verts,
        faces=faces,
        cam=pred_camera,
        angle=90,
        axis=[0,1,0]
    )

    if target_exists:
        result_image = np.hstack([image, pred_image, target_image, render, render_side])
    else:
        result_image = np.hstack([image, pred_image, render, render_side])

    return result_image


def batch_visualize_preds(images, preds, target=None, max_images=16, idxs=None,
                          target_exists=True, dataset='common'):

    if max_images is None or images.shape[0] < max_images:
        max_images = images.shape[0]

    # preds = preds[-1] # get the final output

    with torch.no_grad():
        for k, v in preds.items():
            if isinstance(preds[k], torch.Tensor):
                preds[k] = v.detach().cpu().numpy()
        if target_exists:
            for k, v in target.items():
                if isinstance(target[k], torch.Tensor):
                    target[k] = v.cpu().numpy()

    result_images = []

    indexes = range(max_images) if idxs is None else idxs

    for idx in indexes:
        single_pred = {}
        for k, v in preds.items():
            single_pred[k] = v[idx]

        if target_exists:
            single_target = {}
            for k, v in target.items():
                single_target[k] = v[idx]
        else:
            single_target = None

        img = visualize_preds(images[idx], single_pred, single_target, target_exists,
                              dataset=dataset)
        result_images.append(img)

    result_image = np.vstack(result_images)

    return result_image


def batch_visualize_vid_preds(video, preds, target, max_video=4, vis_hmr=False, dataset='common'):
    with torch.no_grad():
        if isinstance(video, torch.Tensor):
            video = torch_vid2numpy(video) # NTCHW

    video = np.transpose(video, (0, 1, 3, 4, 2))[:max_video]  # NTCHW->NTHWC

    batch_size, tsize = video.shape[:2]

    if vis_hmr:
        features = target['features']
        target_verts, target_cam = get_regressor_output(features)
        target['verts'] = target_verts
        target['cam'] = target_cam

    with torch.no_grad():
        for k, v in preds.items():
            if isinstance(preds[k], torch.Tensor):
                preds[k] = v.cpu().numpy()[:max_video]

        for k, v in target.items():
            if isinstance(target[k], torch.Tensor):
                target[k] = v.cpu().numpy()[:max_video]

    batch_videos = [] # NTCHW*4

    for batch_id in range(batch_size):

        result_video = [] #TCHW*4

        for t_id in range(tsize):
            image = video[batch_id, t_id]
            single_pred = {}
            single_target = {}
            for k, v in preds.items():
                single_pred[k] = v[batch_id, t_id]

            for k, v in target.items():
                single_target[k] = v[batch_id, t_id]

            img = visualize_preds(image, single_pred, single_target,
                                  vis_hmr=vis_hmr, dataset=dataset)

            result_video.append(img[np.newaxis, ...])

        result_video = np.concatenate(result_video)

        batch_videos.append(result_video[np.newaxis, ...])

    final_video = np.concatenate(batch_videos)
    final_video = np.transpose(final_video, (0, 1, 4, 2, 3))  # NTHWC->NTCHW
    return final_video


def draw_skeleton(image, kp_2d, dataset='common', unnormalize=True, thickness=2):

    if unnormalize:
        kp_2d[:,:2] = normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()
    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx,pt in enumerate(kp_2d):
        if pt[2] > 0: # if visible
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
            # cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    for i,(j1,j2) in enumerate(skeleton):
        if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
            if dataset == 'common':
                color = rcolor if common_lr[i] == 0 else lcolor
            else:
                color = lcolor if i % 2 == 0 else rcolor
            pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
            cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    return image


def batch_draw_skeleton(images, target, max_images=8, dataset='common'):
    if max_images is None or images.shape[0] < max_images:
        max_images = images.shape[0]


    with torch.no_grad():

        for k, v in target.items():
            if isinstance(target[k], torch.Tensor):
                target[k] = v.cpu().numpy()

    result_images = []

    for idx in range(max_images):
        single_target = {}

        for k, v in target.items():
            single_target[k] = v[idx]

        img = torch2numpy(images[idx])

        img = draw_skeleton(img.copy(), single_target['kp_2d'], dataset=dataset)
        result_images.append(img)

    result_image = np.vstack(result_images)

    return result_image


def get_regressor_output(features):
    from lib.models.spin import Regressor

    batch_size, seqlen = features.shape[:2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Regressor().to(device)

    smpl = SMPL(SMPL_MODEL_DIR).to(device)
    pretrained = torch.load('models/model_best.pth.tar')['gen_state_dict']

    new_pretrained_dict = {}
    for k, v in pretrained.items():
        if 'regressor' in k:
            new_pretrained_dict[k[10:]] = v
            # adapt mean theta to new batch size
            if 'mean_theta' in k:
                del new_pretrained_dict[k[10:]]

    model.load_state_dict(new_pretrained_dict, strict=False)
    features = features.reshape(batch_size*seqlen, -1)
    features = features.to(device)
    theta = model(features)[-1]

    cam = theta[:, 0:3].contiguous()
    pose = theta[:, 3:75].contiguous()
    shape = theta[:, 75:].contiguous()

    pred_output = smpl(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], pose2rot=True)
    verts = pred_output.vertices # , _, _ = smpl(pose, shape)

    verts = verts.reshape(batch_size, seqlen, -1, 3)
    cam = cam.reshape(batch_size, seqlen, -1)

    return verts, cam

def show_video(video, fps=25):
    for fid, frame in enumerate(video):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(f'frame {fid}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1./fps)

    cv2.destroyAllWindows()
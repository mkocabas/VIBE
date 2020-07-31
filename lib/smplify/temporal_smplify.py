# This script is the extended version of https://github.com/nkolot/SPIN/blob/master/smplify/smplify.py to deal with
# sequences inputs.

import os
import torch

from lib.core.config import VIBE_DATA_DIR
from lib.models.smpl import SMPL, JOINT_IDS, SMPL_MODEL_DIR
from lib.smplify.losses import temporal_camera_fitting_loss, temporal_body_fitting_loss

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

def arrange_betas(pose, betas):
    batch_size = pose.shape[0]
    num_video = betas.shape[0]

    video_size = batch_size // num_video
    betas_ext = torch.zeros(batch_size, betas.shape[-1], device=betas.device)
    for i in range(num_video):
        betas_ext[i*video_size:(i+1)*video_size] = betas[i]

    return betas_ext

class TemporalSMPLify():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 use_lbfgs=True,
                 device=torch.device('cuda'),
                 max_iter=20):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size
        self.max_iter = max_iter
        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=VIBE_DATA_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.use_lbfgs = use_lbfgs
        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.max_iter,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    camera_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints


                    loss = temporal_camera_fitting_loss(model_joints, camera_translation,
                                               init_cam_t, camera_center,
                                               joints_2d, joints_conf, focal_length=self.focal_length)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_camera_fitting_loss(model_joints, camera_translation,
                                           init_cam_t, camera_center,
                                           joints_2d, joints_conf, focal_length=self.focal_length)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints

                    loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()
                # scheduler.step(epoch=i)

        # Get final loss value

        with torch.no_grad():
            betas_ext = arrange_betas(body_pose, betas)
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas_ext)
            model_joints = smpl_output.joints
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation,
                                                           camera_center,
                                                           joints_2d, joints_conf, self.pose_prior,
                                                           focal_length=self.focal_length,
                                                           output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        # Back to weak perspective camera
        camera_translation = torch.stack([
            2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
            camera_translation[:,0], camera_translation[:,1]
        ], dim=-1)

        betas = betas.repeat(pose.shape[0],1)
        output = {
            'theta': torch.cat([camera_translation, pose, betas], dim=1),
            'verts': vertices,
            'kp_3d': joints,
        }

        return output, reprojection_loss
        # return vertices, joints, pose, betas, camera_translation, reprojection_loss

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss

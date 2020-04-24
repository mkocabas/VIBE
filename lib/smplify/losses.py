# This script is the extended version of https://github.com/nkolot/SPIN/blob/master/smplify/losses.py to deal with
# sequences inputs.

import torch
from lib.models.spin import perspective_projection
from lib.models.smpl import JOINT_IDS


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """
    # pose_prior_weight = 1.
    # shape_prior_weight = 1.
    # angle_prior_weight = 1.
    # sigma = 10.

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss
    print(f'joints: {reprojection_loss[0].sum().item():.2f}, '
          f'pose_prior: {pose_prior_loss[0].item():.2f}, '
          f'angle_prior: {angle_prior_loss[0].item():.2f}, '
          f'shape_prior: {shape_prior_loss[0].item():.2f}')

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


def temporal_body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                               joints_2d, joints_conf, pose_prior,
                               focal_length=5000, sigma=100, pose_prior_weight=4.78,
                               shape_prior_weight=5, angle_prior_weight=15.2,
                               smooth_2d_weight=0.01, smooth_3d_weight=1.0,
                               output='sum'):
    """
    Loss function for body fitting
    """
    # pose_prior_weight = 1.
    # shape_prior_weight = 1.
    # angle_prior_weight = 1.
    # sigma = 10.

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    # Smooth 2d joint loss
    joint_conf_diff = joints_conf[1:]
    joints_2d_diff = projected_joints[1:] - projected_joints[:-1]
    smooth_j2d_loss = (joint_conf_diff ** 2) * joints_2d_diff.abs().sum(dim=-1)
    smooth_j2d_loss = torch.cat(
        [torch.zeros(1, smooth_j2d_loss.shape[1], device=body_pose.device), smooth_j2d_loss]
    ).sum(dim=-1)
    smooth_j2d_loss = (smooth_2d_weight ** 2) * smooth_j2d_loss

    # Smooth 3d joint loss
    joints_3d_diff = model_joints[1:] - model_joints[:-1]
    # joints_3d_diff = joints_3d_diff * 100.
    smooth_j3d_loss = (joint_conf_diff ** 2) * joints_3d_diff.abs().sum(dim=-1)
    smooth_j3d_loss = torch.cat(
        [torch.zeros(1, smooth_j3d_loss.shape[1], device=body_pose.device), smooth_j3d_loss]
    ).sum(dim=-1)
    smooth_j3d_loss = (smooth_3d_weight ** 2) * smooth_j3d_loss

    total_loss += smooth_j2d_loss + smooth_j3d_loss

    # print(f'joints: {reprojection_loss[0].sum().item():.2f}, '
    #       f'pose_prior: {pose_prior_loss[0].item():.2f}, '
    #       f'angle_prior: {angle_prior_loss[0].item():.2f}, '
    #       f'shape_prior: {shape_prior_loss[0].item():.2f}, '
    #       f'smooth_j2d: {smooth_j2d_loss.sum().item()}, '
    #       f'smooth_j3d: {smooth_j3d_loss.sum().item()}')

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def temporal_camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                                 focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [JOINT_IDS[joint] for joint in op_joints]
    # gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    # gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    # reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
    #                          projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op).sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()

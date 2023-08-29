import torch
import numpy as np


def quat_to_euler(q, is_degree=True):
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)

    return np.array([roll, pitch, yaw])


def position_dist(pred, target):
    return np.linalg.norm(pred-target, ord=2)


def rotation_dist(pred, target):  # angle-axis

    # Calculate quaternion difference
    # scalar dot product by element-wise multiplying each element, then adding
    quaternion_difference = np.dot(target, pred)

    # Calculate rotation error in radians
    # arc cosine (inverse cosine) is often used in trigonometric calculations, and in this context,
    # it is used to compute the rotation angle based on the quaternion difference.
    alpha = 2 * np.arccos(np.abs(quaternion_difference))

    # Convert radians to degrees
    return alpha * (180.0 / np.pi)


# only for bayesian posenet
def fit_gaussian(pose_quat):
    # pose_quat = pose_quat.detach().cpu().numpy()

    num_data, _ = pose_quat.shape

    # Convert quat to euler
    pose_euler = []
    for i in range(0, num_data):
        pose = pose_quat[i, :3]
        quat = pose_quat[i, 3:]
        euler = quat_to_euler(quat)
        pose_euler.append(np.concatenate((pose, euler)))

    # Calculate mean and variance
    pose_mean = np.mean(pose_euler, axis=0)
    mat_var = np.zeros((6, 6))
    for i in range(0, num_data):
        pose_diff = pose_euler[i] - pose_mean
        mat_var += pose_diff * np.transpose(pose_diff)

    mat_var = mat_var / num_data
    pose_var = mat_var.diagonal()

    return pose_mean, pose_var

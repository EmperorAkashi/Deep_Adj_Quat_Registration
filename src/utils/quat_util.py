import numpy as np
from typing import List
import torch
from scipy.spatial.transform import Rotation as R


def generate_random_rot(max_angle:int) -> np.ndarray:
    """generate a random euler angle from scipy
    the instance "r" will be generated inside the dataset
    """
    curr_angle = R.random().as_euler('zyx', degrees=True)
    scale_factor = 1
    if max_angle != 180:
        scale_factor = max_angle/180
    curr_angle = curr_angle*scale_factor

    return curr_angle

def generate_random_quat(convention:str="Hamitonian") -> np.ndarray:
    """this code generates a random quaternion
    NOTE: this is actually the correct way to do a uniform random rotation in SO3
    """
    
    quat = np.random.uniform(-1, 1, 4)  # note this is a half-open interval, so 1 is not included but -1 is
    norm = np.sqrt(np.sum(quat**2))

    while not (0.2 <= norm <= 1.0):
        quat = np.random.uniform(-1, 1, 4)
        norm = np.sqrt(np.sum(quat**2))
    
    quat = quat / norm

    if convention == "JPL":
        order = [3,0,1,2]
        quat = quat[order]
    return quat

def generate_batch_random_quat(batch:int) -> np.ndarray:
    batch_quat = [generate_random_quat() for _ in range(batch)]
    return np.array(batch_quat)

def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to a rotation matrix."""
    q0, q1, q2, q3 = quat[0], quat[1], quat[2], quat[3]

    matrix = torch.zeros(3, 3, dtype=quat.dtype)

    matrix[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[0, 1] = 2 * (q1 * q2 - q0 * q3)
    matrix[0, 2] = 2 * (q1 * q3 + q0 * q2)

    matrix[1, 0] = 2 * (q1 * q2 + q0 * q3)
    matrix[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[1, 2] = 2 * (q2 * q3 - q0 * q1)

    matrix[2, 0] = 2 * (q1 * q3 - q0 * q2)
    matrix[2, 1] = 2 * (q2 * q3 + q0 * q1)
    matrix[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    return matrix.to(quat.device)

def batch_quat_to_rot(batch_quat: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions to a batch of rotation matrices."""
    batch_rot = [quat_to_rot(q) for q in batch_quat]
    return torch.stack(batch_rot).to(batch_quat.device)


def rot2quat(M:np.ndarray) -> np.ndarray:
    """this fxn manually convert rot mat to the quaternion
    instead of using scipy's Rotation
    """
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q

def batch_rot_to_quat(batch_rot:np.ndarray) -> np.ndarray:
    batch_quat = [quat_to_rot(r) for r in batch_rot]
    return np.array(batch_quat)
import math
import numpy as np
import enum


class ColorModes(enum.Enum):
    NORMAL = 0
    DISTANCE = 1
    UNIFORM = 2


def get_rotation_matrix(orientation):
    angle_x = math.radians(orientation[2])
    angle_y = math.radians(orientation[1])
    angle_z = math.radians(orientation[0])
    rotation_x = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, math.cos(angle_x), -math.sin(angle_x), 0.0],
                           [0.0, math.sin(angle_x), math.cos(angle_x), 0.0], [0.0, 0.0, 0.0, 1.0]])
    rotation_y = np.array(
        [[math.cos(angle_y), 0.0, math.sin(angle_y), 0.0], [0.0, 1.0, 0.0, 0.0],
         [-math.sin(angle_y), 0.0, math.cos(angle_y), 0.0], [0.0, 0.0, 0.0, 1.0]])
    rotation_z = np.array(
        [[math.cos(angle_z), -math.sin(angle_z), 0.0, 0.0], [math.sin(angle_z), math.cos(angle_z), 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    rotation = np.matmul(np.matmul(rotation_x, rotation_y), rotation_z)
    return rotation


def get_translation_matrix(pos):
    translation = np.identity(4)
    translation[0, 3] = pos[0]
    translation[1, 3] = pos[1]
    translation[2, 3] = pos[2]
    return translation


def get_scale_matrix(s):
    scale = np.identity(4)
    scale[0, 0] = s[0]
    scale[1, 1] = s[1]
    scale[2, 2] = s[2]
    return scale


def get_o2w_w2o_matrices(translation, rotation, scale):
    o2w = np.matmul(translation, np.matmul(rotation, scale))
    w2o = np.linalg.inv(o2w)
    return o2w, w2o


def transform_dir(d, transform):
    d_transf = np.copy(d)
    d_transf = np.append(d_transf, [0.0])
    d_transf = np.matmul(transform, d_transf)
    d_transf = d_transf / np.linalg.norm(d_transf)
    d_transf = d_transf[0:3]
    return d_transf

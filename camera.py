import numpy as np
import math
import utils


class Camera:
    def __init__(self):
        self.pos_ = None
        self.orientation_ = None
        self.fov_ = None
        self.near_clip_ = 0.0

        self.up_ = None
        self.right_ = None
        self.forward_ = None

        self.width_ = 0.0
        self.height_ = 0.0

    def load(self, cfg):
        self.pos_ = np.array([cfg['position']['x'], cfg['position']['y'], cfg['position']['z']])
        self.orientation_ = [cfg['orientation']['h'], cfg['orientation']['p'], cfg['orientation']['r']]
        self.fov_ = [cfg['fov_x'], cfg['fov_y']]
        self.near_clip_ = cfg['near_clip']

    def init(self):
        transform = utils.get_rotation_matrix(self.orientation_)

        right = np.array([1.0, 0.0, 0.0])
        right = utils.transform_dir(right, transform)

        up = np.array([0.0, 0.0, 1.0])
        up = utils.transform_dir(up, transform)

        forward = np.cross(up, right)
        forward = forward / np.linalg.norm(forward)

        self.up_ = up
        self.right_ = right
        self.forward_ = forward

        ratio = self.fov_[1] / self.fov_[0]
        self.width_ = 2 * self.near_clip_ * math.tan(math.radians(self.fov_[0]))
        self.height_ = ratio * self.width_

    def get_ray_dir(self, pos2d):
        ray_dir = self.forward_ + self.right_ * ((pos2d[0] - 0.5) * self.width_) + \
                  self.up_ * ((0.5 - pos2d[1]) * self.height_)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        return ray_dir

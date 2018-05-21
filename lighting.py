import numpy as np
import refl_refr


class PointLight:
    def __init__(self, camera_pos, cfg):
        self.pos_ = np.array([cfg['position']['x'], cfg['position']['y'], cfg['position']['z']])
        self.color_ = np.array([cfg['color']['r'], cfg['color']['g'], cfg['color']['b']])
        self.radius_ = cfg['radius']
        self.shininess_ = cfg['shininess']
        self.camera_pos_ = camera_pos

    def illuminate(self, point, normal, K_d, K_s):
        distance = np.linalg.norm(self.pos_ - point)
        if distance > self.radius_:
            res_color = np.zeros(3)
            return res_color
        l = (self.pos_ - point) / distance
        h = (self.camera_pos_ - point) / np.linalg.norm(self.camera_pos_ - point)
        r = refl_refr.get_reflected_ray(-l, normal)
        r = r / np.linalg.norm(r)

        I_diff = np.clip(K_d * max(np.dot(normal, l).astype(float), 0.0), 0.0, 1.0)
        I_spec = np.clip(K_s * (max(np.dot(r, h).astype(float), 0.0) ** self.shininess_), 0.0, 1.0)

        # res_color = (I_diff + I_spec) * self.color_ / (4 * np.pi * distance)
        res_color = (I_diff + I_spec) * self.color_
        return res_color

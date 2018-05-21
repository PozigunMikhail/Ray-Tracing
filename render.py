import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import copy
import refl_refr
import utils


class Render:
    def __init__(self, resolution, trace_depth, scene, color_mode, dist_range=0.0):
        self.resolution_ = resolution
        self.trace_depth_ = trace_depth
        self.camera_ = scene.camera_
        self.scene_ = scene
        self.t_min_ = 1.0
        self.t_max_ = np.inf
        color_res = np.concatenate((resolution, np.array([3])))
        self.canvas_ = np.zeros(color_res)
        np.expand_dims(self.canvas_, axis=-1)
        self.color_mode_ = color_mode
        self.dist_range_ = dist_range

    def trace(self, start_pos, d, trace_depth):
        closest_t = np.inf
        intersection = None
        color = np.copy(self.scene_.background_color_)
        normal = None
        intersected_obj = None
        for obj in self.scene_.objects_:
            res = obj.intersect(start_pos, d, self.color_mode_, self.dist_range_)
            if res is None:
                continue
            col = res[0]
            p = res[1]
            n = res[2]
            t = np.linalg.norm(p - start_pos)
            if self.t_min_ <= t <= self.t_max_ and t < closest_t:
                closest_t = t
                intersection = np.copy(p)
                color = np.copy(col)
                intersected_obj = copy.deepcopy(obj)
                normal = np.copy(n)

        if intersection is None or trace_depth == 0:
            return color, False

        if self.color_mode_ != utils.ColorModes.UNIFORM:
            return color, True

        res_color = np.zeros(3)

        bias = 1.0e-5
        if intersected_obj.material_type_ == refl_refr.MaterialTypes.REFLECTION:
            intersection_biased = intersection + bias * normal
            reflected_ray = refl_refr.get_reflected_ray(d, normal)
            reflected_ray = reflected_ray / np.linalg.norm(reflected_ray)
            reflected_color, is_intersected = self.trace(intersection_biased, reflected_ray,
                                                         trace_depth - 1)
            res_color = (1.0 - intersected_obj.refl_coef_) * color + intersected_obj.refl_coef_ * reflected_color
        elif intersected_obj.material_type_ == refl_refr.MaterialTypes.REFLECTION_AND_REFRACTION:
            is_outside = np.dot(d, normal) > 0.0
            kr, kt = refl_refr.get_fresnel_coeffs(d, normal, intersected_obj.refr_index_)
            if kr < 1.0:
                refracted_ray = refl_refr.get_refracted_ray(d, normal, intersected_obj.refr_index_)
                refracted_ray = refracted_ray / np.linalg.norm(refracted_ray)
                if is_outside is True:
                    refr_start = intersection + bias * normal
                else:
                    refr_start = intersection - bias * normal
                refracted_color, is_intersected = self.trace(refr_start, refracted_ray, trace_depth - 1)
            else:
                refracted_color = np.zeros(3)
            reflected_ray = refl_refr.get_reflected_ray(d, normal)
            intersection_biased = intersection + bias * normal
            reflected_ray = reflected_ray / np.linalg.norm(reflected_ray)
            reflected_color, is_intersected = self.trace(intersection_biased, reflected_ray, trace_depth - 1)
            res_color = (1 - 0.6) * color + 0.6 * (kr * reflected_color + kt * refracted_color)
        elif intersected_obj.material_type_ == refl_refr.MaterialTypes.DIFFUSE:
            light_color = np.zeros(3)
            for light in self.scene_.light_sources_:
                intersection_biased = intersection + bias * normal
                shadow_ray = (light.pos_ - intersection_biased) / np.linalg.norm(light.pos_ - intersection_biased)
                c, is_intersected = self.trace(intersection_biased, shadow_ray, 1)
                if is_intersected is True:
                    shadow = 0.0
                else:
                    shadow = 1.0
                light_color = light_color + shadow * light.illuminate(intersection, normal, intersected_obj.K_d_,
                                                                      intersected_obj.K_s_)
            res_color = np.clip(color + light_color, 0.0, 1.0)
        return np.clip(res_color, 0.0, 1.0), True

    def render(self):
        for i in range(self.resolution_[1]):
            for j in range(self.resolution_[0]):
                x = float(j) / self.resolution_[0]
                y = float(i) / self.resolution_[1]
                ray_dir = self.camera_.get_ray_dir([x, y])
                color, is_intersected = self.trace(self.camera_.pos_, ray_dir, self.trace_depth_)
                self.canvas_[i, j, :] = color
                # self.canvas_[self.canvas_ < 0.0] = 0.0

    def draw(self):
        pass
        # imgplot = plt.imshow(self.canvas_)
        # plt.show()

    def save(self, filename="result.bmp"):
        # image = Image.open("res.bmp")
        im = Image.fromarray((255 * self.canvas_).astype(np.uint8))
        im.save(filename)

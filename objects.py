import numpy as np
import math
import utils
import refl_refr


class Primitive:
    def __init__(self, cfg):
        orientation = np.array([cfg['lcs']['h'], cfg['lcs']['p'], cfg['lcs']['r']])
        pos = np.array([cfg['lcs']['x'], cfg['lcs']['y'], cfg['lcs']['z']])
        scale = np.array([cfg['lcs']['sx'], cfg['lcs']['sy'], cfg['lcs']['sz']])
        rotation = utils.get_rotation_matrix(orientation)
        translation = utils.get_translation_matrix(pos)
        scale = utils.get_scale_matrix(scale)
        o2w, w2o = utils.get_o2w_w2o_matrices(translation, rotation, scale)
        self.o2w_ = np.copy(o2w)
        self.w2o_ = np.copy(w2o)
        # self.o2w_tr_ = np.copy(np.linalg.inv(self.o2w_.transpose()))
        self.o2w_tr_ = self.o2w_.transpose()

        self.color_ = np.array(
            [cfg['material']['color']['r'], cfg['material']['color']['g'], cfg['material']['color']['b']])
        self.material_type_ = refl_refr.MaterialTypes.DIFFUSE
        if ('type' in cfg['material']) is True:
            if cfg['material']['type'] == 'REFLECTION':
                self.material_type_ = refl_refr.MaterialTypes.REFLECTION
            elif cfg['material']['type'] == 'REFLECTION_AND_REFRACTION':
                self.material_type_ = refl_refr.MaterialTypes.REFLECTION_AND_REFRACTION
        self.refr_index_ = 1.0
        if ('refr_index' in cfg['material']) is True:
            self.refr_index_ = cfg['material']['refr_index']
        self.refl_coef_ = 0.5
        if ('refl_coef' in cfg['material']) is True:
            self.refl_coef_ = cfg['material']['refl_coef']
        self.K_d_ = 0.3
        if ('K_d' in cfg['material']) is True:
            self.K_d_ = cfg['material']['K_d']
        self.K_s_ = 0.5
        if ('K_s' in cfg['material']) is True:
            self.K_s_ = cfg['material']['K_s']

    def ray2object(self, start_pos, d):
        s = np.copy(start_pos)
        r = np.copy(d)

        s = np.append(s, [1.0])
        r = np.append(r, [0.0])
        s = np.matmul(self.w2o_, s)
        s = s / s[3]
        r = np.matmul(self.w2o_, r)
        # r = r / r[3]
        return s[0:3], r[0:3]

    def point2world(self, point):
        p = np.copy(point)
        p = np.append(p, [1.0])
        p = np.matmul(self.o2w_, p)
        p = p / p[3]
        return p[0:3]

    def normal2world(self, normal):
        n = np.copy(normal)
        n = np.append(n, [0.0])
        n = np.matmul(self.o2w_tr_, n)
        return n[0:3]

    def get_all_intersections(self, start_pos, d):  # returns list of (point, normal)
        return []

    def intersect(self, start_pos, d, color_mode, dist_range=0):
        s, r = self.ray2object(start_pos, d)
        r = r / np.linalg.norm(r)
        intersections = self.get_all_intersections(s, r)
        closest_t = np.inf
        intersection = None
        for elem in intersections:
            t = np.linalg.norm(elem[0] - start_pos)
            if t < closest_t:
                closest_t = t
                intersection = np.copy(elem)
        if intersection is None:
            return None
        p = self.point2world(intersection[0])
        n = self.normal2world(intersection[1])
        n = n / np.linalg.norm(n)

        color = None
        if color_mode == utils.ColorModes.NORMAL:
            color = np.abs(n)
        elif color_mode == utils.ColorModes.DISTANCE:
            col = np.linalg.norm(p - start_pos)
            color = col / dist_range
        elif color_mode == utils.ColorModes.UNIFORM:
            color = self.color_
        return [color, p, n]


class Sphere(Primitive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rad_ = cfg['sphere']['r']

    def get_all_intersections(self, start_pos, d):
        rad = self.rad_
        # d = dir
        k1 = np.dot(d, d)  # == 1
        k2 = 2.0 * np.dot(start_pos, d)
        k3 = np.dot(start_pos, start_pos) - rad ** 2
        discr = k2 ** 2 - 4 * k1 * k3
        if discr >= 0:
            t1 = (-k2 + math.sqrt(discr)) / 2 / k1
            t2 = (-k2 - math.sqrt(discr)) / 2 / k1
            result = []
            if t1 > 0.0:
                result.append((start_pos + t1 * d, self.get_normal(start_pos + t1 * d)))
            if t2 > 0.0:
                result.append((start_pos + t2 * d, self.get_normal(start_pos + t2 * d)))
            return result
        return []

    def get_normal(self, p):
        n = np.array([2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])
        return n / np.linalg.norm(n)


class Plane(Primitive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.width_ = cfg['plane']['width']
        self.height_ = cfg['plane']['height']
        self.normal_ = np.array([0.0, 0.0, 1.0])

    def get_all_intersections(self, start_pos, d):
        # d = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        denom = np.dot(d, self.normal_)
        if np.abs(denom) > 1.0e-5:
            t = np.dot(-start_pos, self.normal_) / denom
            if t < 0.0:
                return []
            p = start_pos + t * d
            if p[0] > 0.5 * self.width_ \
                    or p[0] < -0.5 * self.width_ \
                    or p[1] > 0.5 * self.height_ \
                    or p[1] < -0.5 * self.height_:
                return []
            return [(p, self.normal_)]
        return []

    def get_normal(self, p):
        return self.normal_


class Cylinder(Primitive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rad_ = cfg['cylinder']['r']
        self.height_ = cfg['cylinder']['h']

    def intersect_with_disks(self, start_pos, d):
        result = []
        rad = self.rad_
        positions = [np.array([0.0, 0.0, 0.5 * self.height_]), np.array([0.0, 0.0, -0.5 * self.height_])]
        # d = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        normals = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
        for i in range(len(positions)):
            denom = np.dot(d, normals[i])
            if np.abs(denom) <= 1.0e-5:
                return []
            t = np.dot(positions[i] - start_pos, normals[i]) / denom
            if t < 0.0:
                continue
            p = start_pos + t * d
            if (p[0] - positions[i][0]) ** 2 + (p[1] - positions[i][1]) ** 2 <= rad ** 2:
                result.append((p, np.copy(normals[i])))
        return result

    def intersect_with_cylinder(self, start_pos, d):
        rad = self.rad_
        # d = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        k1 = d[0] ** 2 + d[1] ** 2
        k2 = 2.0 * d[0] * start_pos[0] + 2.0 * d[1] * start_pos[1]
        k3 = start_pos[0] ** 2 + start_pos[1] ** 2 - rad ** 2
        discr = k2 ** 2 - 4 * k1 * k3
        if discr >= 0:
            t = []
            p = []
            t.append((-k2 + math.sqrt(discr)) / 2 / k1)
            t.append((-k2 - math.sqrt(discr)) / 2 / k1)
            p.append(start_pos + t[0] * d)
            p.append(start_pos + t[1] * d)
            result = []
            for i in [0, 1]:
                if -0.5 * self.height_ <= p[i][2] <= 0.5 * self.height_ and t[i] > 0.0:
                    result.append((p[i], self.get_normal_for_cylinder(p[i])))
            return result
        return []

    def get_all_intersections(self, start_pos, d):
        return self.intersect_with_cylinder(start_pos, d) + self.intersect_with_disks(start_pos, d)
        # return self.intersect_with_disks(start_pos, ray)

    def get_normal_for_cylinder(self, p):
        # return np.array([1.0, 0.0, 0.0])
        n = np.array([2.0 * p[0], 2.0 * p[1], 0.0])
        return n / np.linalg.norm(n)


class Cone(Primitive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rad_ = cfg['cone']['r']
        self.height_ = cfg['cone']['h']

    def intersect_with_disk(self, start_pos, d):
        # d = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        normal = np.array([0.0, 0.0, -1.0])
        denom = np.dot(d, normal)
        if np.abs(denom) > 1.0e-5:
            t = np.dot(-start_pos, normal) / denom
            if t < 0.0:
                return []
            p = start_pos + t * d
            if p[0] ** 2 + p[1] ** 2 <= self.rad_ ** 2:
                return [(p, normal)]
        return []

    def intersect_with_cone(self, start_pos, d):
        ratio = self.rad_ / self.height_
        top = np.array([0.0, 0.0, self.height_])
        # d = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        k1 = d[0] ** 2 + d[1] ** 2 - (ratio * d[2]) ** 2
        k2 = 2.0 * d[0] * (start_pos[0] - top[0]) + 2.0 * d[1] * (start_pos[1] - top[1]) - 2.0 * (ratio ** 2) * d[2] * (
            start_pos[2] - top[2])
        k3 = (start_pos[0] - top[0]) ** 2 + (start_pos[1] - top[1]) ** 2 - (ratio * (start_pos[2] - top[2])) ** 2
        discr = k2 ** 2 - 4 * k1 * k3
        if discr >= 0:
            t = []
            p = []
            t.append((-k2 + math.sqrt(discr)) / 2 / k1)
            t.append((-k2 - math.sqrt(discr)) / 2 / k1)
            p.append(start_pos + t[0] * d)
            p.append(start_pos + t[1] * d)
            result = []
            for i in [0, 1]:
                if 0.0 <= p[i][2] <= self.height_ and t[i] > 0.0:
                    result.append((p[i], self.get_normal_for_cone(p[i])))
            return result
        return []

    def get_all_intersections(self, start_pos, d):
        return self.intersect_with_disk(start_pos, d) + self.intersect_with_cone(start_pos, d)

    def get_normal_for_cone(self, p):
        n = np.array([2.0 * p[0], 2.0 * p[1], -2.0 * p[2] * (self.rad_ / self.height_) ** 2])
        return n / np.linalg.norm(n)
        # return np.array([1.0, 0.0, 0.0])


class Triangle(Primitive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.v0_ = np.array([cfg['triangle']['x1'], cfg['triangle']['y1'], cfg['triangle']['z1']])
        self.v1_ = np.array([cfg['triangle']['x2'], cfg['triangle']['y2'], cfg['triangle']['z2']])
        self.v2_ = np.array([cfg['triangle']['x3'], cfg['triangle']['y3'], cfg['triangle']['z3']])
        normal = np.cross(self.v1_ - self.v0_, self.v2_ - self.v0_)

        self.normal_ = normal / np.linalg.norm(normal)
        self.d_ = np.dot(self.normal_, self.v0_)

    def get_all_intersections(self, start_pos, d):
        # r = (ray - start_pos) / np.linalg.norm(ray - start_pos)
        denom = np.dot(d, self.normal_)
        if np.abs(denom) < 1.0e-5:
            return []
        t = -(np.dot(self.normal_, start_pos) + self.d_) / denom
        if t < 0.0:
            return []
        p = start_pos + t * d
        edge0 = self.v1_ - self.v0_
        vp0 = p - self.v0_
        c0 = np.cross(edge0, vp0)
        if np.dot(self.normal_, c0) < 0:
            return []

        edge1 = self.v2_ - self.v1_
        vp1 = p - self.v1_
        c1 = np.cross(edge1, vp1)
        if np.dot(self.normal_, c1) < 0:
            return []

        edge2 = self.v0_ - self.v2_
        vp2 = p - self.v2_
        c2 = np.cross(edge2, vp2)
        if np.dot(self.normal_, c2) < 0:
            return []
        return [(p, self.normal_)]

    def get_normal(self, p):
        return self.normal_

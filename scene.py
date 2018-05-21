import numpy as np
import objects
import lighting


class Scene:
    def __init__(self, camera):
        self.camera_ = camera
        self.objects_ = []
        self.light_sources_ = []
        self.background_color_ = np.array([1.0, 1.0, 1.0])

    def load(self, cfg):
        for scene_obj in cfg:
            if ('node' in scene_obj) is True:
                node = scene_obj['node']
                if ('sphere' in node) is True:
                    self.objects_.append(objects.Sphere(scene_obj['node']))
                elif ('cylinder' in node) is True:
                    self.objects_.append(objects.Cylinder(scene_obj['node']))
                elif ('cone' in node) is True:
                    self.objects_.append(objects.Cone(scene_obj['node']))
                elif ('plane' in node) is True:
                    self.objects_.append(objects.Plane(scene_obj['node']))
                elif ('triangle' in node) is True:
                    self.objects_.append(objects.Triangle(scene_obj['node']))
            elif ('light' in scene_obj) is True:
                light = scene_obj['light']
                self.light_sources_.append(lighting.PointLight(self.camera_.pos_, light))

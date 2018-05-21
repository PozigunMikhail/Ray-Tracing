from optparse import OptionParser
import numpy as np
import camera
import scene
import render
import utils
import yaml

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--scene", dest="scene",
                      help="Scene", default="scene.yml")
    parser.add_option("--resolution_x", type="int", dest="resolution_x", default=200)
    parser.add_option("--resolution_y", type="int", dest="resolution_y", default=200)
    parser.add_option("-o", "--output", dest="output",
                      help="Output", default="result2.bmp")
    parser.add_option("--trace_depth", type="int", dest="trace_depth", default=2)
    parser.add_option("--normal_as_color", action="store_true", dest="normal_as_color", default=False)
    parser.add_option("--distance_as_color", action="store_true", dest="distance_as_color", default=False)
    parser.add_option("--dist_range", type="int", dest="dist_range", default=30)
    (options, args) = parser.parse_args()
    # print(options)

    with open(options.scene, 'r', encoding='utf8') as yaml_file:
        cfg = yaml.load(yaml_file)

    camera = camera.Camera()
    camera.load(cfg['camera'])
    camera.init()

    scene = scene.Scene(camera)
    scene.load(cfg['scene'])

    color_mode = utils.ColorModes.UNIFORM
    if options.normal_as_color is True:
        color_mode = utils.ColorModes.NORMAL
    elif options.distance_as_color is True:
        color_mode = utils.ColorModes.DISTANCE
    render = render.Render(np.array([options.resolution_x, options.resolution_y]), options.trace_depth, scene,
                           color_mode, options.dist_range)
    render.render()
    # render.draw()
    render.save(options.output)

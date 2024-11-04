#import matplotlib.pyplot as plt
from PIL import Image,ImageTk
import io
import datetime
import gphoto2 as gp
import subprocess
import tkinter as tk
import os
import time
from PIL import ImageDraw,ImageFont


def setCameraConfig(camera,configname,configvalue):
    gp.check_result(gp.gp_camera_init(camera))
    # get configuration tree
    config = gp.check_result(gp.gp_camera_get_config(camera))
    # find the capture target config item
    capture_target = gp.check_result(
        gp.gp_widget_get_child_by_name(config, str(configname)))
    # check value in range
    count = gp.check_result(gp.gp_widget_count_choices(capture_target))

    value = int(configvalue)

    if value < 0 or value >= count:
        print('Parameter out of range')
        return 1
    # set value
    value = gp.check_result(gp.gp_widget_get_choice(capture_target, value))
    gp.check_result(gp.gp_widget_set_value(capture_target, value))
    # set config
    gp.check_result(gp.gp_camera_set_config(camera, config))
    return 0


def init(value):
    camera = gp.check_result(gp.gp_camera_new())
    gp.check_result(gp.gp_camera_init(camera))
    # get configuration tree
    config = gp.check_result(gp.gp_camera_get_config(camera))
    # find the capture target config item
    capture_target = gp.check_result(
        gp.gp_widget_get_child_by_name(config, 'output'))
    # check value in range
    count = gp.check_result(gp.gp_widget_count_choices(capture_target))
    if value < 0 or value >= count:
        print('Parameter out of range')
        return 1
    # set value
    value = gp.check_result(gp.gp_widget_get_choice(capture_target, value))
    gp.check_result(gp.gp_widget_set_value(capture_target, value))
    # set config
    gp.check_result(gp.gp_camera_set_config(camera, config))
    # clean up
    time.sleep(5)
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    gp.check_result(gp.gp_camera_exit(camera))
    return 0

def initCaptureTargets():
    # open camera connection
    camera = gp.check_result(gp.gp_camera_new())
    gp.check_result(gp.gp_camera_init(camera))
    # get configuration tree
    config = gp.check_result(gp.gp_camera_get_config(camera))
    # find the capture target config item
    capture_target = gp.check_result(
        gp.gp_widget_get_child_by_name(config, 'capturetarget'))
    # print current setting
    value = gp.check_result(gp.gp_widget_get_value(capture_target))
    print('Current setting:', value)
    # print possible settings
    for n in range(gp.check_result(gp.gp_widget_count_choices(capture_target))):
        choice = gp.check_result(gp.gp_widget_get_choice(capture_target, n))
        print('Choice:', n, choice)
    # clean up
    gp.check_result(gp.gp_camera_exit(camera))
    return 0

def cameraConfig():
    context = gp.Context()
    camera = gp.Camera()
    camera.init(context)
    config_tree = camera.get_config(context)
    print('=======')

    total_child = config_tree.count_children()
    for i in range(total_child):
        child = config_tree.get_child(i)
        text_child = '# ' + child.get_label() + ' ' + child.get_name()
        print(text_child)

        for a in range(child.count_children()):
            grandchild = child.get_child(a)
            text_grandchild = '    * ' + grandchild.get_label() + ' -- ' + grandchild.get_name()
            print(text_grandchild)

            try:
                text_grandchild_value = '        Setted: ' + grandchild.get_value()
                print(text_grandchild_value)
                print('        Possibilities:')
                for k in range(grandchild.count_choices()):
                    choice = grandchild.get_choice(k)
                    text_choice = '         - ' + choice
                    print(text_choice)
            except:
                pass
            print()
        print()

    camera.exit(context)

cameraConfig()
value = 0
init(value)


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


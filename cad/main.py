# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cadquery as cq
import math
import numpy as np
from cadquery import cqgi

#cq-editor started from conda env cadquery-editor with cq-editor --> open .py file that is loaded in pycharm --> autoreload


debug = True
# These can be modified rather than hardcoding values for each dimension.
length = 120.0  # Length of the block
height = 40.0  # Height of the block
thickness = 10.0  # Thickness of the block
center_hole_dia = 12.0  # Diameter of center hole in block
padding = 10.0

hex_dia = 10.0 #length of the hexagon (circle creating it)
hex_internal_h = round((hex_dia/2)*math.sin(math.pi/3),3)
print(hex_internal_h)

hex_outer_distance = hex_dia/4
hex_vertical_distance = 2*hex_internal_h+hex_outer_distance
#hex_angled_distance =
hex_angled_dx = hex_vertical_distance *math.cos(math.pi/6)
hex_angled_dy = hex_vertical_distance *math.sin(math.pi/6)
print(hex_vertical_distance)


hex_per_column = 5
x_off = 0.0
y_off = 0.0
print(int(hex_per_column/2))
hex_points = np.empty(hex_per_column, dtype=object)
for index in range(hex_per_column):

    i = index
    print(i)
    if i > int(hex_per_column/2):
        factor = 1
        i = int(hex_per_column / 2)-i
    else:
        factor = 1

    new_x = x_off
    new_y = y_off + (i*hex_vertical_distance*factor)
    print(new_y)
    hex_points[index] = (new_x,new_y)
    #temp = np.array([(new_x,new_y)])
    #hex_points = np.concatenate((hex_points, temp), axis=0)
print(hex_points)

# Create a block based on the dimensions above and add a 22mm center hole.
# 1.  Establishes a workplane that an object can be built on.
# 1a. Uses the X and Y origins to define the workplane, meaning that the
# positive Z direction is "up", and the negative Z direction is "down".
# 2.  The highest (max) Z face is selected and a new workplane is created on it.
# 3.  The new workplane is used to drill a hole through the block.
# 3a. The hole is automatically centered in the workplane.

result_test1 = (
    cq.Workplane("XY").box(length, height, thickness)
        .faces(">Z").workplane()
        .hole(center_hole_dia)
        .faces(">Z").workplane()
        .rect(30,30,forConstruction=True)
        .vertices()
        .hole(3)
)

width , length , height = 180.0 , 70.0 , 5.0

#    .pushPoints([(0,0),(0,hex_vertical_distance),(0.0,hex_vertical_distance*2),(hex_angled_dx,hex_angled_dy),(2*hex_angled_dx,2*hex_angled_dy)])
result_test2 = (
    cq.Workplane("XY").box(width, length, height)
    .pushPoints(hex_points)
    .polygon(6,hex_dia).cutThruAll()
)

result = result_test2

if debug:
    try:
        show_object(result)
        #debug(result)
    except:
        pass
else:
    result.val().exportStep('obj.step')

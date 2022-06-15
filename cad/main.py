# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cadquery as cq
import math
import numpy as np
from cadquery import cqgi
from PIL import ImageFont #pip install PILLOW



#cq-editor started from conda env cadquery-editor with cq-editor --> open .py file that is loaded in pycharm --> autoreload


debug = True
# These can be modified rather than hardcoding values for each dimension.
base_length = 120.0  # Length of the block
base_height = 40.0  # Height of the block
thickness = 10.0  # Thickness of the block
center_hole_dia = 12.0  # Diameter of center hole in block
padding = 5.0

base_width , base_length , base_height = 180.0 , 70.0 , 5.0


hex_dia = 11.0 #length of the hexagon (circle creating it)
hex_internal_h = round((hex_dia/2)*math.sin(math.pi/3),3)
print(hex_internal_h)

hex_outer_distance = hex_dia/4
hex_vertical_distance = 2*hex_internal_h+hex_outer_distance
#hex_angled_distance =
hex_angled_dx = hex_vertical_distance *math.cos(math.pi/6)
hex_angled_dy = hex_vertical_distance *math.sin(math.pi/6)
print(hex_vertical_distance)


hex_per_column = 7
columns = int((base_width - hex_outer_distance)/hex_dia) + 2
print(columns)
x_off = 0.0
y_off = 0.0
print(int(hex_per_column/2))
hex_points = np.empty(hex_per_column*columns, dtype=object)
for column_index in range(columns):
    j = column_index
    x_off = column_index*hex_angled_dx
    if column_index > int((columns-1) / 2):
        j = int((columns-1) / 2) - j
        x_off = j*hex_angled_dx
    if j%2 == 0:
        y_off = 0.0
    else:
        y_off = hex_angled_dy
    print(j)
    for index in range(hex_per_column):
        #create a column of hex that are placed vertically on top of each other
        #distance between outerpoints of the hexagon is "hex_vertical_distance"
        i = index
        #print(i)
        if i > int(hex_per_column/2):
            factor = 1
            i = int(hex_per_column / 2)-i

        new_x = x_off
        new_y = y_off + (i*hex_vertical_distance)
        #print(new_y)
        hex_points[index+column_index*hex_per_column] = (new_x,new_y)
        #temp = np.array([(new_x,new_y)])
        #hex_points = np.concatenate((hex_points, temp), axis=0)
print(hex_points)

padding_points = [(base_width/2,base_length/2),
                  (base_width/2,-1*(base_length/2)),
                  (-1*base_width/2,base_length/2),
                  (-1*base_width/2,-1*(base_length/2)),
                  ((base_width / 2) - padding, (base_length / 2)-padding),
                  ((base_width / 2) - padding, -1 * (base_length / 2) + padding),
                  (padding+(-1 * base_width / 2), (base_length / 2)-padding,
                   (padding+(-1 * base_width / 2), padding + (-1 * (base_length / 2)))),
                  ]
padding_points = [(60,60),(-60,-60),
                  (-60,50),(60,-60),
                  (50,50),
                  (50,-50),
                  (-50,50),
                  (-50,-50)]
def calculateStringSize(font,str):
    str_size = font.getsize(str)
    print(str_size)
    str_size = (str_size[0] * 1.2, str_size[1] * 0.78)
    print(str_size)
    return str_size
names_str = "Marion&Manfjred"
bigText_str = "50ahre"
font_size=32
font = ImageFont.truetype("C:/Windows/Fonts/PER_____.ttf",font_size)
names_str_size = calculateStringSize(font,names_str)
font_size_bigText=60
font = ImageFont.truetype("C:/Windows/Fonts/PER_____.ttf",font_size_bigText)
bigText_str_size = calculateStringSize(font,bigText_str)
# Create a block based on the dimensions above and add a 22mm center hole.
# 1.  Establishes a workplane that an object can be built on.
# 1a. Uses the X and Y origins to define the workplane, meaning that the
# positive Z direction is "up", and the negative Z direction is "down".
# 2.  The highest (max) Z face is selected and a new workplane is created on it.
# 3.  The new workplane is used to drill a hole through the block.
# 3a. The hole is automatically centered in the workplane.

result_test1 = (
    cq.Workplane("XY").box(base_width, base_length, base_height)
    .faces(">Z").workplane()
    .text("25Jahre",25,-3,combine='a',kind="bold")
    #.box(8, 8, 8).faces(">Z").workplane().text("Z", 5, -1.0)
        #.hole(center_hole_dia)
        #.faces(">Z").workplane()
        #.rect(30,30,forConstruction=True)
        #.vertices()
        #.hole(3)
)



result_test2 = (
    cq.Workplane("XY").box(base_width, base_length, base_height)
    .faces(">Z").workplane()
    .pushPoints(hex_points)
    .polygon(6, hex_dia).cutThruAll()
    .faces("<Z").workplane()
    .rect(base_width - 2 * padding, base_length - 2 * padding)
    .rect(base_width, base_length)
    .extrude(-base_height)
    .faces("<Z").workplane(origin=(0, -base_length/3.2, 0))
    .text(bigText_str, 60, -10, cut=False, combine=True, font="Perpetua", kind="bold").mirror()
    .faces("<Z").workplane(origin=(0, base_length / 3, 0))
    .text(names_str, font_size, -10, cut=False, combine=True, font="Perpetua", kind="bold")
    .faces(">Z").workplane(origin=(0, base_length / 3, 0))
    .rect(names_str_size[0],names_str_size[1]).extrude(-2)
    .faces(">Z").workplane(origin=(0, -base_length/3.2, 0))
    .rect(bigText_str_size[0],bigText_str_size[1]).extrude(-2)
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

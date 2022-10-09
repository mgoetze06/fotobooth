from PIL import Image, ImageDraw, ImageFilter, ImageOps
#from PIL import ImagePath
#import os
import random
import glob

def readHexPointsFromCSV(filename):
    f = open(filename, 'r')
    hex_points = []
    lines = f.readlines()
    # print("got %d lines from %s with header: "%(len(lines),filename))
    # print(lines[0].split(";"))
    for line in lines:
        if not (line[0] == 'i'):
            line = line.strip().split(";")
            # print(line[5])
            hex_points = hex_points + [(int(line[1]) ,int(line[2]))]
    print(hex_points)
    return hex_points
def placeMultipleHexagon(background ,img ,position ,dia ,outline ,rot):
    # drawing a mask as a hexagon, then applying this mask to the img and pasting this img on the background
    # dia = hex diameter
    # rot = hex rotation (0 for flat side up, 90 for tip up)
    # outline = drawing an outline as a factor of masksize around the hexagon (only using green at the moment)
    # position = multiple tuple of (x,y)

    # background: single path to image from background
    # img: list of filenames that contain the used faces

    # position and img list need to have same size!
    f_outl = 1.06
    masksize = int(2*dia)
    back = Image.open(background).resize((1920 ,1080)).convert("RGB")

    mask = Image.new(mode="L" ,size = (masksize ,masksize), color = (0))  # (mode = "L",
    draw = ImageDraw.Draw(mask)
    draw.regular_polygon(bounding_circle=((int(masksize /2),int(masksize /2),dia)), n_sides=6, rotation=rot, fill=(255))

    if outline > 1:
        # f_outl = 1.06
        f_outl = outline
        outline_mask = Image.new(mode="L" ,size = (int(f_outl *masksize) ,int(f_outl *masksize)), color = (0))  # (mode = "L",
        outline_draw = ImageDraw.Draw(outline_mask)
        outline_draw.regular_polygon(bounding_circle=((int((f_outl *masksize) /2) ,int((f_outl *masksize ) /2) ,f_outl *dia)),n_sides=6, rotation=90, fill=(255))
        img_outline = Image.open('green.png').resize((int(f_outl*masksize),int(f_outl*masksize))).convert("RGB")

    for index,point in enumerate(position):
        filename = img[index]
        print(filename)
        if outline > 1:
            back.paste(img_outline ,(int(point[0 ] -f_outl *dia) ,int(point[1 ] -f_outl *dia)) ,outline_mask)

        img_face = Image.open(filename).resize((masksize ,masksize)).convert("RGB")
        back.paste(img_face ,(int(point[0 ] -dia) ,int(point[1 ] -dia)) ,mask)

    # plt.imshow(background)

    # cv2.imwrite("collage.jpg", background)
    # background = background.save("selection_collage.jpg")
    return back



def detectOnImg():
    print("")
def detectionImgToCollage(img_dir,collage_dir):

    #faces = glob.glob("detections/*jpg")[:5]
    faces = random.choices(glob.glob("detections/*jpg"),k=5)
    # print(faces)
    # print(big_points)
    print(len(faces))
    big_points = readHexPointsFromCSV("big_hex_locations.csv")
    collage = placeMultipleHexagon('background.jpg', faces, big_points[:len(faces)], 255, 1.06, 90)
    collages = len(glob.glob(collage_dir+"/*jpg"))
    collage = collage.save(collage_dir + "/hex_collage_" + str(collages) + ".jpg")
    #cv2.imshow("window",collage)

detectionImgToCollage("detections","collages")
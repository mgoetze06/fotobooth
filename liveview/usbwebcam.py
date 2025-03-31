import cv2
import tkinter as tk
from PIL import Image, ImageTk

import RPi.GPIO as GPIO
import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import time
import os, random, shutil
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106
import _rpi_ws281x as ws
import subprocess
from subprocess import check_output
import psutil
from gpiozero import CPUTemperature
from datetime import datetime
import cv2
import io
import gphoto2 as gp
from webserver.fotobooth_utils import writeImagecountToFile,writeCollageCountToFile,readRGBFromFile,IsCustomCollageEnabled


def resizeImageToCanvas(pilImage,w,h):
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)

        try:
            r,g,b = readRGBFromFile()
            new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(r,g,b))
            new_img.paste(pilImage, (round(scr_w/2-imgWidth/2),0))
            pilImage = new_img
        except:
            print("error setting background")
    return pilImage

def convert_to_tkimage(frame):
    # Convert the frame from BGR to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Convert to ImageTk format
    return ImageTk.PhotoImage(image=pil_image)

if __name__ == '__main__':


    
    root = tk.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    #root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (w, h))
    root.overrideredirect(False)
   
    root.persistent_image = None
    root.attributes('-fullscreen',True)
    root.configure(background='black')
    
    #root.focus_set()    
    root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
    canvas = tk.Canvas(root,width=w,height=h,highlightthickness=0)
   
    canvas.pack()
    canvas.configure(background='black')
    root.update()
    
    scr_w = 1920
    scr_h = 1080
    try:
        pilImage = Image.open('/home/pi/programs/countdown/picwait.jpg')
    except:
        pass

    pilImage = resizeImageToCanvas(pilImage,w,h)

    #updateCanvas(pilImage,root,canvas)
    image = ImageTk.PhotoImage(pilImage)
    imagesprite = canvas.create_image(w/2,h/2,image=image)
    root.update()
    newimage = False
    cap = cv2.VideoCapture(cv2.CAP_V4L2)

    livecanvas = tk.Canvas(canvas, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    livecanvas.pack()
    i = 0
    while i < 100:
        ret, frame = cap.read()
        
        if ret:
            # Convert image from OpenCV BGR format to RGB
            photo = convert_to_tkimage(frame)
            
            # Update the canvas with the new image
            livecanvas.create_image(0, 0, image=photo, anchor=tk.NW)
            print("updated image")

        root.update()
        i += 1
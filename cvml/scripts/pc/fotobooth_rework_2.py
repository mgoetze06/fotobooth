from threading import Thread, Lock, Event
import time
import cv2
import numpy as np
import os
import glob
import random
import multiprocessing
from multiprocessing.managers import BaseManager


def rescaleImage(img,w,h):
    rescaled_image = np.zeros((h, w, 3), np.uint8)
    imgHeight, imgWidth, _ = img.shape
    #if imgWidth >= w or imgHeight >= h:
    ratio = min(w / imgWidth, h / imgHeight)
    # ratio = w / imgWidth
    # ratio = h / imgHeight
    imgWidth = int(imgWidth * ratio)
    imgHeight = int(imgHeight * ratio)
    #print(imgWidth, imgHeight)
    # pilImage = pilImage.resize((imgWidth, imgHeight), Image.ANTIALIAS)
    resized = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)
    x_offset = int(w / 2 - imgWidth / 2)
    y_offset = int(h / 2 - imgHeight / 2)
    #print(x_offset, y_offset)
    rescaled_image[y_offset:y_offset + imgHeight, x_offset:x_offset + imgWidth] = resized
    #cv2.imwrite("resized.jpg", resized)
    return rescaled_image

def listImages(img_folder):
    #global img_folder
    images = []
    if not img_folder:
        dir = os.getcwd()
    else:
        dir = img_folder[:-1]
    print(dir)
    for file in glob.glob(dir + "/*.JPG"):
    #for file in os.listdir(dir):
        if file.endswith(".jpg") or file.endswith(".JPG"):
            images.append(file)
    #imglist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
    imglist = sorted(images, key=os.path.getmtime)
    #print(imglist)
    return imglist

def trackHistory(img):
    global historyLenght, historyImg

    if historyImg == []:
        for i in range(historyLenght):
            historyImg.append(None)
        #print("history init: ", self.historyImg)
    for i in range(historyLenght-1,0,-1):
        if i > 0:
            historyImg[i] = historyImg[i-1]
    historyImg[0] = img


def handleFrame(event,fotobooth,mouse_clicked):
    global cv2windowinit
    #mouse_clicked_bool = False

    def mouse_click(event, x, y, flags, param):
        global mouse_clicked_bool, new_image_to_display
        name = "[########## mouse click]: "
        #mouse_clicked_bool = False
        print(name,mouse_clicked.is_set())
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(name + "status before mouseclick " + str(status))
            # status = 2
            with lock:
                mouse_clicked.set()
            #new_image_to_display.clear()
            # newImg = False
            #mouse_clicked_bool = True
            print(name + "mouse button pressed")
            # print(name +"status after mouseclick " + str(status))
# mouse_clicked_bool
    while True:
    #while not self.stopThread:
        #print("I'm handling myself")
        #print("[process] new_image_to_display is set to: ", event.is_set())
        if event.is_set() and not mouse_clicked.is_set():
            print("[process] trying to update screen")
            if not cv2windowinit:
                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow("window", 40, 30)
                cv2.setMouseCallback("window", mouse_click, param=(mouse_clicked))
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2windowinit = True
            # cv2.imshow("window", img)
            #print("[process] value of mouseCallback: ",mouse_clicked_bool)
            img = fotobooth.getFrame()
            path = fotobooth.getFramePath()
            print("[process] recieved image path: ",path)
            print("mouse clicked: ",mouse_clicked.is_set())
            if not mouse_clicked.is_set():
                cv2.imshow("window", img)
                k = cv2.waitKey(1) & 0xFF
                #k = cv2.waitKey(3)
                print("[process] update screen done")
                event.clear()
            if k == ord('q'):
                #exit
                break;
        else:
            time.sleep(0.001)
    print("done handling")
    return

def handleStatus():
    global gallery_time,status,historyImg,img_after_collage,new_image_to_display,img_folder
    while True:
        if mouse_clicked.is_set():
            status = 2
        name = "[handleStatus]: "
        print(name + str(status))
        if status == 0:
            print("gallery image")
            #normal image for gallery
            images = listImages(img_folder)
            imgpath = random.choice(images)
            while imgpath in historyImg:
                imgpath = img_folder + random.choice(images)
            image = cv2.imread(imgpath)
            image = rescaleImage(image, 1920, 1080)

            if img_after_collage > 4:
                img_after_collage = 0
                status = 1

        elif status == 1:
                print("collage image")
                #random collage if collages exist already
                imgpath = img_folder + "selection_collage.jpg"
                image = cv2.imread(imgpath)
                status = 0
                print("collage image")
        elif status == 2:
            if mouse_clicked.is_set():
                print("animate to new image")
                #animate to new image
                imgpath = img_folder + "fotobooth_wait.JPG"
                image = cv2.imread(imgpath)
                image = rescaleImage(image, 1920, 1080)
                cv2.putText(image, "animate to new image", (100, 100), 1, 2, (255, 255, 255), 2)
                mouse_clicked.clear()
                status = 3
        elif status == 3:
                print("new image")
                #display new image
                imgpath = img_folder + "test_gesicht.JPG"
                image = cv2.imread(imgpath)
                image = rescaleImage(image, 1920, 1080)
                cv2.putText(image,"new image",(100,100),1,2,(255,255,255),2)

                status = 0
        elif status == 100:
            imgpath = "empty imagepath"
            print("idling")

        if status != 100:
            try:
                #print("writing new image to fotobooth handler")
                if mouse_clicked.is_set() and status < 2:
                    print("skipping gallery image due to mouse clicked")
                else:
                    print(imgpath)
                    fotobooth.setFrame(image,imgpath)
                    #print("done writing")
                    new_image_to_display.set()
                    print(new_image_to_display.is_set())
            except:
                print(imgpath)
                print("not able to set image to fotobooth handler")


        start = time.time()
        while time.time() - start < gallery_time:
            if mouse_clicked.is_set():
                status = 2
                break
            #if self.stopThread:
            #    return
            #if self.userinput:
            #    break
            #else:
            time.sleep(0.2)

        print(name + "handle status loop done, new status: ",status)

class FotoboothHandler:
    def __init__(self):
        self.imgpath = "test_gesicht.jpg"
        self.frame = cv2.imread(self.imgpath)
        self.started = False
        #self.thread = None
        #self.threadImage = None
        #self.img_after_newimg = 0
        #self.img_after_collage = 0
        #self.stopThread = False
        #self.newImg = False
        #self.status = 0
        #self.userinput = Event()
        #self.historyLenght = 7
        #self.historyImg = []#np.empty(shape=(self.historyLenght,1))
        #self.cv2windowinit = False



    def start(self):
        if self.started:
            return
        else:
            self.started = True
        return

    def stop(self):
        self.started = False

    def setFrame(self,img,imgpath):
        #global new_image_to_display
        #print("imgpath in setframe")
        #print(imgpath)
        if self.started:
            img = rescaleImage(img,1920,1080)
            #with lock:
                #if not self.userinput:
            self.frame = img
            self.imgpath = imgpath
                #new_image_to_display.set()
        #print("Setframe is done")
        return

    def getFrame(self):
        if self.started:
            return self.frame

    def getFramePath(self):
        if self.started:
            return self.imgpath

        #print(self.historyImg)



gallery_time = 3
historyImg = []
historyLenght = 7
newImg = cv2windowinit = False
lock = multiprocessing.Lock()
img_after_collage = 0
status = 0

img_folder = "C:/projects/fotobooth/cvml/notebooks"
if not img_folder.endswith("/"):
    img_folder = img_folder + "/"
#fotobooth = FotoboothHandler()

if __name__ == '__main__':
    new_image_to_display = multiprocessing.Event()
    mouse_clicked = multiprocessing.Event()


    BaseManager.register('FotoboothHandler', FotoboothHandler)
    manager = BaseManager()
    manager.start()
    fotobooth = manager.FotoboothHandler()

    fotobooth.start()
    firstimage = img_folder + "test_gesicht.jpg"
    fotobooth.setFrame(cv2.imread(firstimage),firstimage)


    process_handle_frame = multiprocessing.Process(name='process_handle_frame', target=handleFrame, args=(new_image_to_display,fotobooth,mouse_clicked))  #handle new frame
    process_handle_frame.daemon = True
    process_handle_frame.start()



    while True:
        handleStatus()
        print("main:")
        print(mouse_clicked.is_set())
    #fotobooth.getNextImage(True)
    #time.sleep(2)
    #fotobooth.getNextImage(False)
    #time.sleep(2)
    #fotobooth.stop()


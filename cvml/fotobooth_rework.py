from threading import Thread, Lock
import time
import cv2
import numpy as np
import os
import glob
import random

gallery_time = 2
lock = Lock()

def rescaleImage(img,w,h):
    rescaled_image = np.zeros((h, w, 3), np.uint8)
    imgHeight, imgWidth, _ = img.shape
    #if imgWidth >= w or imgHeight >= h:
    ratio = min(w / imgWidth, h / imgHeight)
    # ratio = w / imgWidth
    # ratio = h / imgHeight
    imgWidth = int(imgWidth * ratio)
    imgHeight = int(imgHeight * ratio)
    print(imgWidth, imgHeight)
    # pilImage = pilImage.resize((imgWidth, imgHeight), Image.ANTIALIAS)
    resized = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)
    x_offset = int(w / 2 - imgWidth / 2)
    y_offset = int(h / 2 - imgHeight / 2)
    print(x_offset, y_offset)
    rescaled_image[y_offset:y_offset + imgHeight, x_offset:x_offset + imgWidth] = resized
    #cv2.imwrite("resized.jpg", resized)
    return rescaled_image

class FotoboothHandler:
    def __init__(self):
        self.frame = None
        self.started = False
        self.thread = None
        self.threadImage = None
        self.img_after_newimg = 0
        self.stopThread = False
        self.newImg = False
        self.status = 0
        self.userinput = False
        self.historyLenght = 5
        self.historyImg = np.array(self.historyLenght)


    def start(self):
        if self.started:
            return
        self.handleFramethread = Thread(target=self.handleFrame, args=())
        self.handleFramethread.start()
        self.handleStatusthread = Thread(target=self.handleStatus, args=())
        self.handleStatusthread.start()
        self.started = True
        return self

    def stop(self):
        if self.started:
            self.stopThread = True
            self.started = False

    def handleFrame(self):
        while not self.stopThread:
            print("I'm handling myself")
            if self.newImg:
                self.newImg = False
                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # cv2.imshow("window", img)
                cv2.imshow("window", self.getFrame())
                k = cv2.waitKey(5) & 0xFF
                if k == ord('q'):
                    self.stop()
                    break;
            else:
                time.sleep(0.1)
        print("done handling")
        return
    def getFrame(self):
        return self.frame

    def trackHistory(self,img):
        for i  in range(self.historyLenght,1,-1):
            print("index: ",i)
            if i > 0:
                self.historyImg[i] = self.historyImg[i-1]
        self.historyImg[0] = img
        print(self.historyImg)

        #TODO List index out of range for history image array

    def listImages(self):
        images = []
        for file in os.listdir(os.getcwd()):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                images.append(file)
        #imglist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
        imglist = sorted(images, key=os.path.getmtime)
        #print(imglist)
        return imglist

    def handleStatus(self):
        global gallery_time
        while not self.stopThread:
            print("handling status")
            match self.status:
                case 0:
                    #normal image for galleryq
                    images = self.listImages()
                    img_name = random.choice(images)
                    print(img_name)
                    image = cv2.imread(img_name)
                    image = rescaleImage(image, 1920, 1080)
                case 1:
                    #random collage if collages exist already
                    image = cv2.imread("selection_collage.jpg")
                case 2:
                    #display new image
                    image = cv2.imread("test_gesicht.JPG")
                    image = rescaleImage(image, 1920, 1080)
                    cv2.putText(image,"new image",(100,100),1,1,(255,255,255),2)
            with lock:
                self.frame = image
                self.newImg = True
                self.trackHistory(img_name)
            #wait for next image to be displayed
            start = time.time()
            while time.time() - start < gallery_time:
                if self.stopThread:
                    return
                if self.userinput:
                    break
                else:
                    time.sleep(0.1)
                    #print(time.time() - start)
            #self.status += 1
            #if self.status > 2:
            #    self.status = 0
        return

fotobooth = FotoboothHandler()
fotobooth.start()
#fotobooth.getNextImage(True)
#time.sleep(2)
#fotobooth.getNextImage(False)
#time.sleep(2)
#fotobooth.stop()


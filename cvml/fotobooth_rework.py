from threading import Thread, Lock
import time
import cv2

lock = Lock()

class FotoboothHandler:
    def __init__(self):
        self.frame = None
        self.started = False
        self.thread = None
        self.threadImage = None
        self.img_after_newimg = 0
        self.stopThread = False
        self.newImg = False

    def start(self):
        if self.started:
            return
        self.thread = Thread(target=self.handleFrame, args=())
        self.thread.start()
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
                    break;
        print("done handling")
        self.stopThread = False
        return
    def getFrame(self):
        return self.frame

    def getNextImage(self,other):
        print("getting next image")
        if other:
            image = cv2.imread("selection_collage.jpg")
        else:
            image = cv2.imread("collage.jpg")
        with lock:
            self.frame = image
            self.newImg = True
        #time.sleep(2)
        return
fotobooth = FotoboothHandler()
fotobooth.start()
fotobooth.getNextImage(True)
time.sleep(2)
fotobooth.getNextImage(False)
time.sleep(2)
fotobooth.stop()


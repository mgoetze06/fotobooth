import numpy as np
import cv2
from threading import Thread, Lock
import time # not really needed, used to simulate the 2 seconds of generation

lock = Lock()

class ImageGenerator:
    def __init__(self, src=0):
      # initialize it with zeros to always have something to show. You can set it to None and check it outside before displaying
      self.frame = np.zeros((100,100,1))
      self.stopped = True
      self.first = True

    def start(self):
     # checks if the generator is still running, to avoid two threads doing the same
     if not self.stopped:
       return
     self.stopped = False
     #Launches a thread to update itself
     Thread(target=self.update, args=()).start()
     return self

    def update(self):
      # go until stop is called, you can set other criterias
      while True:
        if self.stopped:
          return
        # generate the image, this is equal to finalimage = generate_image() in your code

        image = np.random.randint(0,255, (600, 800, 3), dtype=np.uint8)

        if self.first:
            image = cv2.imread("collage.jpg")
            self.first = False
        else:
            image = cv2.imread("selection_collage.jpg")
            self.first = True
        # this sleep is to simulate that it took longer to execute
        time.sleep(2)
        with lock:
          self.frame = image

    # if this changes the other thread will stop
    def stop(self):
      self.stopped = True

    # gets latest frame available
    def get(self):
      return self.frame

# creates the object and start generating
imGen = ImageGenerator()
imGen.start()
# infinite loop to display the image, it can be stopped at any point with 'q'
while (True):
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("window", img)
    cv2.imshow("window", imGen.get())
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
      break;
# stops the generator and the other thread
imGen.stop()
cv2.destroyAllWindows()
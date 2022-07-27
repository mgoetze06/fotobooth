import cv2
import numpy as np


w = 1920
h = 1080
blank_image = np.zeros((h,w,3), np.uint8)

img = cv2.imread("test_gesicht.JPG")


def rescaleImage(img,w,h):
    rescaled_image = np.zeros((h, w, 3), np.uint8)
    imgHeight, imgWidth, _ = img.shape
    if imgWidth > w or imgHeight > h:
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
        cv2.imwrite("resized.jpg", resized)
    return rescaled_image

img = rescaleImage(img,w,h)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("window", img)
cv2.waitKey(3000)
img = cv2.imread("selection_collage.jpg")
cv2.imshow("window", img)
cv2.waitKey(0)




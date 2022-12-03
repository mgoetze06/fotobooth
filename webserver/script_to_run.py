import time
import cv2

print("I'm a new script")
img = cv2.imread("/home/boris/projects/fotobooth/1.jpg")
print(img.shape[0])
print(img.shape[1])
print(img.shape[2])
factor = 3
img = cv2.resize(img,(int(img.shape[1]/factor),int(img.shape[0]/factor)))
cv2.imshow("Image",img)
cv2.waitKey()
cv2.destroyAllWindows()
#while(True):
#    time.sleep(1)
#    print("running")
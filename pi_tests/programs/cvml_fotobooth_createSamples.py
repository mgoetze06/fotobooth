#!/usr/bin/env python
# coding: utf-8

import mediapipe as mp
import gphoto2 as gp
import os
import io
import datetime
import time
import numpy as np
import cv2

def printSummary(camera):
    #camera = gp.Camera()
    #camera.init()
    #camera.wait_for_event(100)
    camera.get_config()
    text = camera.get_summary()
    print('Summary')
    print('=======')
    print(str(text))
    #camera.exit()


def takePhoto(camera):
    #camera = gp.Camera()
    #camera.init()
    print('Capturing image')
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('.', file_path.name)
    print('Copying image to', target)
    camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    #camera.exit()

def getPreview(cam):
    a = time.time()
    camera_file = gp.check_result(gp.gp_camera_capture_preview(cam))
    file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
    #data = memoryview(file_data)
    gp.check_result(gp.gp_camera_exit(camera))
    #camera.capture(gp.GP_CAPTURE_IMAGE)
    image_io = io.BytesIO(file_data)
    #image = Image.open(image_io)
    #print(image.size)
    b = time.time()
    print("processed preview in %s ms"%(b-a))
    return image_io


def takeMultiplePreviews(camera,previews):
    #camera = gp.Camera()
    #camera.wait_for_event(100)
    image_counter = 0
    start = datetime.datetime.now()
    while image_counter < previews:
        #camera.get_config()
        image,image_io = getPreview(camera)
        #image.show()
        #print(image_io)
        #print(image_io)
        image_io.seek(0)
        org_img = cv2.imdecode(np.frombuffer(image_io.read(), np.uint8), 1)
        filename = str(image_counter)+".jpg"
        cv2.imwrite(filename,org_img)
        image_counter = image_counter + 1
    end = datetime.datetime.now()
    print("average processing time %s ms"%(((end-start).total_seconds()*1000)/image_counter))


if __name__ == '__main__':
    camera = gp.Camera()
    camera.init()
    camera.wait_for_event(100)

    printSummary(camera)
    time.sleep(2)

    image_io = getPreview(camera)
    print(image_io)
    image_io.seek(0)
    org_img = cv2.imdecode(np.frombuffer(image_io.read(), np.uint8), 1)
    cv2.imshow("window",org_img)
    cv2.waitKey()

    #takeMultiplePreviews(camera,10)

    camera.exit()
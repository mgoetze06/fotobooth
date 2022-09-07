import cv2
import numpy as np
import time
import mediapipe as mp
import tensorflow as tf
#from tensorflow.keras.models import load_model
import math
from testmodel import processLandmarks

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

def readImg(path,rescale_res):
    #org_img = cv2.imread("C:/projects/fotobooth/data/rohdaten/pos/im20.JPG")
    #org_img = cv2.imread("C:/projects/fotobooth/data/rohdaten/pos/im19.JPG")
    org_img = cv2.imread(path)
    h_org,w_org,temp = org_img.shape
    #print(org_img.shape)
    if rescale_res == None:
        img = org_img
    else:
        img = cv2.resize(org_img, rescale_res)
    #plt.figure(figsize=(6, 4), dpi=150)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(im_gray, cmap="gray")
    #print(im_gray.shape)
    reducer = org_img.shape[0] / im_gray.shape[0] ,org_img.shape[1] / im_gray.shape[1]
    print("reading image completed: ",path,"\t resolution: ",im_gray.shape)
    return im_gray,reducer,org_img


def detectOnImg(img,scale,neighbors,classifier):

    face_boxes,rejectLevels, levelWeights = classifier.detectMultiScale3(img, scaleFactor=scale, minNeighbors=neighbors, outputRejectLevels=True)
    #print(face_boxes)
    #print(rejectLevels)
    #print(levelWeights)
    print("detection completed")
    return face_boxes,rejectLevels,levelWeights


def drawBoxes(function_img, boxes, weights, score_val, only_max_confidence):
    new_img = np.zeros((function_img.shape[0], function_img.shape[1]), np.uint8)
    new_img[:, :] = function_img[:, :]
    del function_img
    # score_val = 1

    #print(boxes)
    #print(weights)
    #plt.clf()
    #plt.figure(figsize=(6, 4), dpi=150)
    i = d_boxes = 0
    verified_boxes = []
    if only_max_confidence:

        #print(weights)
        index = np.argmax(weights)
        (x, y, w, h) = boxes[index]
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        d_boxes = 1
        verified_boxes += [(x, y, w, h)]
    else:

        for (x, y, w, h) in boxes:
            # print(x, y, w, h)

            if (weights[i] > score_val):
                #print("drawing on img: ", x, y, w, h)
                cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0),
                              3)  # cv2.rectangle(org_img,(int(x*reduce_factor), int(y*reduce_factor)),(int(x*reduce_factor + w*reduce_factor), int(y*reduce_factor + h*reduce_factor)), (0, 255, 0), 5)
                #print("drawing done")
                d_boxes += 1
                verified_boxes += [(x, y, w, h)]
            i += 1
    # plt.imshow(new_img, cmap="gray")
    print("drawing on test image completed")
    return new_img, d_boxes, verified_boxes

def drawBoxOnOrigImg(org_img,reduce_factor,ver_boxes):
    i = 0
    #print(drawn_boxes)
    #fig, axs = plt.subplots(drawn_boxes, figsize=(25, 25))
    if len(ver_boxes) > 0:
        for face_box in ver_boxes:
            # x, y, w_box, h_box = face_box*reduce_factor
            x, w_box = face_box[0] * reduce_factor[1], face_box[2] * reduce_factor[1]
            y, h_box = face_box[1] * reduce_factor[0], face_box[3] * reduce_factor[0]
            # h_org of original image
            # w_org of original image
            #print(x, y, w_box, h_box)
            h_org, w_org, temp = org_img.shape
            crop_padding = int(h_org / 20)  # crop border is 1/15 of the orig. image height
            x = x - crop_padding
            y = y - crop_padding
            if x <= 0:
                x = 0
            if y <= 0:
                y = 0

            # create size of face box with padding
            w_box = w_box + 2 * crop_padding
            h_box = h_box + 2 * crop_padding

            # if face is near edge --> cropped image is moved from edge away
            if (x + w_box) >= w_org:
                x = w_org - w_box
            if (y + h_box) >= h_org:
                y = h_org - h_box

            cv2.rectangle(org_img, (int(x), int(y)), (int(x + w_box), int(y + h_box)), (0, 255, 0), 3)
            # print(x, y, w_box, h_box)
            #cropped_img = org_img[int(y):int(y + h_box), int(x):int(x + w_box)]

            # plt.imshow(cropped_img)
            #new_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            #filename = "glasses%s.jpg" % (axs_index)
            #cv2.imwrite(filename, cropped_img)
            #if drawn_boxes > 1:
            #    axs[axs_index].imshow(new_cropped_img)
            #
            #    axs_index = axs_index + 1
            #else:
            #    plt.imshow(new_cropped_img)
            i = i + 1
    print("drawing on original image completed")
    return org_img

def eval_glassesClassifier(show):
    starttime = time.time()
    # try:#
    #
    #   del im_gray
    #   del im_rect
    # except:
    #    pass
    # im_rect = im_gray
    h, w = 2592, 3888
    resolutions = []

    for i in range(11):
        res_h = int(h * i / 10)
        res_w = int(w * i / 10)
        resolutions.append([(res_w, res_h)])
    if show:
        print("resolutions")
        print(resolutions)
    total_time = 0
    pics_processed = 0
    fastest = 100
    no_success = []
    diff = 0
    for i in range(len(resolutions)):
        if i > 0:
            # print(resolutions[i][0])
            score_val = 1.5
            show_only_max_confidence = True
            start = time.time()
            glasses_classifier = cv2.CascadeClassifier(
                r"big_glasses_classifier_5.xml")
            im_gray, reducer, org_img = readImg("test_gesicht2.JPG", resolutions[i][0])
            # for i in range(2):
            face_boxes, rejectLevels, levelWeights = detectOnImg(im_gray, 1.8, 5, glasses_classifier)
            if len(levelWeights) > 0:
                im_test, drawn_boxes, ver_boxes = drawBoxes(im_gray, face_boxes, levelWeights, score_val,
                                                            show_only_max_confidence)
                org_img = drawBoxOnOrigImg(org_img, reducer, ver_boxes)
                org_img = rescaleImage(org_img, 1920, 1080)
                pics_processed += 1
                end = time.time()
                diff = end - start
                if diff < fastest:
                    winner = resolutions[i][0]
                    fastest = diff
            else:
                no_success.append(resolutions[i][0])
                org_img = rescaleImage(org_img, 1920, 1080)
            if show:
                cv2.imshow("window", org_img)
                print("resolution: ", resolutions[i])
                print(diff)
            total_time += diff
            if show:
                k = cv2.waitKey() & 0xFF
    endtime = time.time()
    print()
    print("######################################")
    print("test results:")

    print("totaltime: \t\t\t",endtime-starttime)
    print("pics with success: \t\t", pics_processed)
    print("avail. pics: \t\t\t", len(resolutions) - 1)
    print("average time per pic: \t\t", total_time / pics_processed)
    print("average fps: \t\t\t", 1 / (total_time / pics_processed))
    print("fastest res: \t\t\t", winner, fastest)
    print("no success on res: \t\t", no_success)
    print("######################################")

def eval_facesClassifier(show):
    starttime = time.time()

    #h, w = 2592, 3888
    #h, w = 648, 972
    h, w = 1292, 1944
    resolutions = []

    for i in range(11):
        res_h = int(h * i / 10)
        res_w = int(w * i / 10)
        resolutions.append([(res_w, res_h)])
    if show:
        print("resolutions")
        print(resolutions)
        print(len(resolutions))
    total_time = 0
    pics_processed = 0
    fastest = 100
    no_success = []
    diff = 0
    print("How many faces are on the picture?")
    number_of_faces = input("faces:")
    #print(number_of_faces)
    incorrect_detection = 0
    detection_correct = False
    for i in range(len(resolutions)):
        if i > 0:
            # print(resolutions[i][0])
            score_val = 4
            show_only_max_confidence = False
            start = time.time()
            face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            im_gray, reducer, org_img = readImg("test_gesicht3.JPG", resolutions[i][0])
            # for i in range(2):
            face_boxes, rejectLevels, levelWeights = detectOnImg(im_gray,1.1, 15, face_cas)
            if len(levelWeights) > 0:
                im_test, drawn_boxes, ver_boxes = drawBoxes(im_gray, face_boxes, levelWeights, score_val,
                                                            show_only_max_confidence)
                #print(drawn_boxes)
                detection_correct = (int(drawn_boxes) == int(number_of_faces))
                if detection_correct:
                    print(number_of_faces," faces were found (",drawn_boxes," boxes were drawn)")
                    pics_processed += 1
                else:
                    no_success.append(resolutions[i][0])
                    incorrect_detection += 1
                org_img = drawBoxOnOrigImg(org_img, reducer, ver_boxes)
                org_img = rescaleImage(org_img, 1920, 1080)
                end = time.time()
                diff = end - start
                if diff < fastest and detection_correct:
                    winner = resolutions[i][0]
                    fastest = diff
            else:
                no_success.append(resolutions[i][0])
                incorrect_detection += 1
                org_img = rescaleImage(org_img, 1920, 1080)
            if show:
                cv2.imshow("window", org_img)
                print("resolution: ", resolutions[i])
                print(diff)
            total_time += diff
            if show:
                k = cv2.waitKey() & 0xFF
            del org_img
    endtime = time.time()
    print()
    print("######################################")
    print("test results:")
    print("totaltime: \t\t",endtime-starttime)
    print("pics with success: \t",pics_processed,"/",len(resolutions) - 1)
    print("average time per pic: \t", total_time / (pics_processed+incorrect_detection))
    print("average fps: \t\t", 1 / (total_time / (pics_processed+incorrect_detection)))
    print("fastest res: \t\t", winner, fastest)
    print("no success on res: \t", no_success)
    print("incorrect detections: \t",incorrect_detection,"/",len(resolutions) - 1)
    print("######################################")

def handDetection(thumbs_img,debug):
    mpHands = mp.solutions.hands
    print(mpHands)
    #hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
    hands = mpHands.Hands(static_image_mode=True,max_num_hands=2, min_detection_confidence=0.4)
    #hands = mpHands.Hands()
    print(hands)
    mpDraw = mp.solutions.drawing_utils
    print(mpDraw)
    print("mediapipe init done")
    # Load the gesture recognizer model
    
    #model = load_model('/home/pi/cvml/programs/hand-gesture-recognition-code/mp_hand_gesture')
    
    #model = load_model('mp_hand_gesture')
    # Load class name
    f = open('/home/pi/cvml/programs/hand-gesture-recognition-code/gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    if debug:
        print(classNames)
    thumbs_img = cv2.imread(thumbs_img)
    # thumbs_img = cv2.imread('thumbsup.jpg')
    # thumbs_img = cv2.imread('thumbsdown.jpg')
    # thumbs_img = cv2.imread('thumbsdown.jpg')
    # thumbs_img = cv2.imread('face0.jpg')

    h_org, w_org, temp = thumbs_img.shape
    if debug:
        print(thumbs_img.shape)
    #reduce_factor = 6
    #thumbs_img = cv2.resize(thumbs_img, (int(w_org / reduce_factor), int(h_org / reduce_factor)))
    thumbs_img = cv2.resize(thumbs_img, (600,400))

    # thumbs_img = cv2.GaussianBlur(thumbs_img, (5,5), 0)
    framergb = cv2.cvtColor(thumbs_img, cv2.COLOR_BGR2RGB)
    #framergb = thumbs_img
    #plt.clf()
    #plt.imshow(framergb)
    #cv2.imshow("window", thumbs_img)
    #cv2.waitKey()
    x, y, c = framergb.shape
    if debug:
        print(framergb.shape)

    # Get hand landmark prediction
    print("mediapipe processing frame")
    result = hands.process(framergb)
    #print(result.multi_hand_landmarks)
    className = ''
    # post process the result
    if result.multi_hand_landmarks:
        print("found hand landmarks")
        landmarks_pred = []
        landmarks_draw = []
        shape = thumbs_img.shape
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks_pred.append([lmx, lmy])
                landmarks_draw.append([lm.x * shape[1], lm.y * shape[0]])
                cv2.circle(thumbs_img, (int((lmx / x) * y), int((lmy / y) * x)), 15, (255, 0, 0))
            mpDraw.draw_landmarks(thumbs_img, handslms, mpHands.HAND_CONNECTIONS)
    else:
        if debug:
            print("no hand landmarks")
    # Predict gesture in Hand Gesture Recognition project
    outarray = []
    if result.multi_hand_landmarks:
        print("found landmarks")
        for i in range(len(landmarks_pred) // 21):
            # predict hand class with model
            prediction = processLandmarks([landmarks_pred[i * 21:(21 + (i * 21))]])
            #prediction = model.predict([landmarks_pred[i * 21:(21 + (i * 21))]])
            # prediction = model.predict([landmarks_pred[0:21]])
            # --> landmarks sind dann doppelt so lang wie sie für eine hand sein müssten
            pred = prediction.flatten()
            classID = np.argmax(pred[0:len(pred) - 1])  # ignore smile
            className = classNames[classID]
            sorted_ind = np.unravel_index(np.argsort(pred.ravel()),pred.shape)
            second_classID = sorted_ind[0][-2]
            if debug:
                print(prediction)
                print(pred)
                print(pred[classID])
                print("model prediction: %s" % className)
                print(second_classID)
                print("second largest pred: ",classNames[second_classID])
                print(int(landmarks_draw[i * 21][0]),int(landmarks_draw[i * 21][1]))
            #first prediction
            cv2.putText(thumbs_img, className, (int(landmarks_draw[4+i * 21][0]),int(landmarks_draw[4+i * 21][1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
            cv2.putText(thumbs_img, str(round(pred[classID],4)), (int(landmarks_draw[4+i * 21][0]+10),int(landmarks_draw[4+i * 21][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
            #second prediction
            cv2.putText(thumbs_img, classNames[second_classID], (int(landmarks_draw[4+i * 21][0]),int(landmarks_draw[4+i * 21][1]-90)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 55), 1,cv2.LINE_AA)
            cv2.putText(thumbs_img, str(round(pred[second_classID],4)), (int(landmarks_draw[4+i * 21][0]+10),int(landmarks_draw[4+i * 21][1]-75)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 55), 1,cv2.LINE_AA)
            outarray.append([className,str(round(pred[classID],4)),classNames[second_classID],str(round(pred[second_classID],4))])
        #cv2.imshow("window", thumbs_img)
        #cv2.waitKey()
    else:
        
        #if debug:
        print("no hand landmarks in img")

    return outarray,thumbs_img

def handDetectionHandler():
    print("detecting hands on an image")
    print("model is not preloaded!")
    #TODO hier setup function einfügen, die das model lädt und an handDetection() übergibt

    starttime = time.time()
    detection,thumbs_img = handDetection('thumbsdown.jpg',debug=True)
    print("found",len(detection), "hand(s)")
    print(detection)
    cv2.imshow("window", thumbs_img)
    endtime = time.time()
    print("detection time: ",endtime -starttime)
    cv2.waitKey()

if __name__ == '__main__':

    #test different resolutions on the same picture
    #eval_glassesClassifier(show=False)
    #eval_facesClassifier(show=False)

    #test hand detection
    handDetectionHandler()




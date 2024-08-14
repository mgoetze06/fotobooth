import cv2
import numpy as np
import sys



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


def drawBoxes(function_img, boxes, weights, score_val, only_max_confidence,draw_rectangle):
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
        if draw_rectangle:
            cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        d_boxes = 1
        verified_boxes += [(x, y, w, h)]
    else:

        for (x, y, w, h) in boxes:
            # print(x, y, w, h)

            if (weights[i] > score_val):
                #print("drawing on img: ", x, y, w, h)
                if draw_rectangle:
                    cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0),3)  # cv2.rectangle(org_img,(int(x*reduce_factor), int(y*reduce_factor)),(int(x*reduce_factor + w*reduce_factor), int(y*reduce_factor + h*reduce_factor)), (0, 255, 0), 5)
                #print("drawing done")
                d_boxes += 1
                verified_boxes += [(x, y, w, h)]
            i += 1
    # plt.imshow(new_img, cmap="gray")
    print("drawing on test image completed")
    return new_img, d_boxes, verified_boxes

def drawBoxOnOrigImg(org_img,reduce_factor,ver_boxes,draw_rectangle):
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

            if draw_rectangle:
                cv2.rectangle(org_img, (int(x), int(y)), (int(x + w_box), int(y + h_box)), (0, 255, 0), 3)
            # print(x, y, w_box, h_box)
            cropped_img = org_img[int(y):int(y + h_box), int(x):int(x + w_box)]

            # plt.imshow(cropped_img)
            new_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            filename = "detections/face_%s.jpg" % (i)
            print(filename)
            cv2.imwrite(filename, cropped_img)
            #if drawn_boxes > 1:
            #    axs[axs_index].imshow(new_cropped_img)
            #
            #    axs_index = axs_index + 1
            #else:
            #    plt.imshow(new_cropped_img)
            i = i + 1
    print("drawing on original image completed")
    return org_img

def cascadeClassifierHandler(img_path,classifier):
    score_val = 4.2
    show_only_max_confidence = False
    #classifier = cv2.CascadeClassifier(r"big_glasses_classifier_5.xml")
    #classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classifier = cv2.CascadeClassifier(classifier)

    im_gray, reducer, org_img = readImg(img_path, (1296,862))
    # for i in range(2):
    face_boxes, rejectLevels, levelWeights = detectOnImg(im_gray, 1.01, 15, classifier)
    if len(levelWeights) > 0:
        im_test, drawn_boxes, ver_boxes = drawBoxes(im_gray, face_boxes, levelWeights, score_val,show_only_max_confidence,draw_rectangle=False)
        #cv2.imshow("window", im_gray)
        #k = cv2.waitKey() & 0xFF
        org_img = drawBoxOnOrigImg(org_img, reducer, ver_boxes,draw_rectangle=False)
        #cv2.imshow("window", org_img)
        #k = cv2.waitKey() & 0xFF
        #org_img = rescaleImage(org_img, 1920, 1080)

    #else:
    #    no_success.append(resolutions[i][0])
    #    org_img = rescaleImage(org_img, 1920, 1080)
    #if show:
    #    cv2.imshow("window", org_img)
    #    print("resolution: ", resolutions[i])
    #    print(diff)

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='Script to detect faces or big glasses.')
    #parser.add_argument("--imagepath", type=str, default="")
    #parser.add_argument("--classifier", type=str, default="faces")
    print("script started")
    #args = parser.parse_args()
    imagepath = sys.argv[1]

    classifier = sys.argv[2]
    #imagepath = args.imagepath
    #classifier = args.classifier
    if classifier == "face":
        classifier = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    else:
        classifier = r"big_glasses_classifier_5.xml"
    cascadeClassifierHandler(imagepath,classifier)
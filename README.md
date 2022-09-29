# Fotobooth
Fotobooth is a python script running on a raspberry pi, controlling a Canon DSLR 1000D through gphoto2. Users can trigger the camera through a footswitch.

<img src="https://github.com/mgoetze06/fotobooth/blob/main/overview.png?raw=true" width="75%">
<img src="https://github.com/mgoetze06/fotobooth/blob/main/1.jpg?raw=true" width="50%">

![CAD](/cad/) Files for the background lights and nerdfacts sign can be found [here](/cad/).

<img src="https://github.com/mgoetze06/fotobooth/blob/main/2.jpg?raw=true" width="50%">
<img src="https://github.com/mgoetze06/fotobooth/blob/main/3.jpg?raw=true" width="25%">


# Computer Vision and Machine Learning
The fotobooth was part of an educational project at the Leipzig University of Applied Sciences (HTWK Leipzig). The folder [cvml](/cvml/) contains all scripts and notebooks developed during the project. The whole project relies heavily on opencv for python.
>https://pypi.org/project/opencv-python/

The cvml-project consists of two seperate parts:
1) hand detection and recognition
2) face (and object) detection

## Mediapipe hand detection
The project uses the mediapipe framework from google to detect hand landmarks on an image. It returns an 1x21x2 Array containing the position of finger joints and finger tips in the image.

MEDIAPIPE LINK TO THUMBS UP

## Tensorflow Gesture Recognition
The landmarks that were detected by mediapipe are processed by a tensorflow lite model to recognize the gestures "thumbs up" and "thumbs down". The model was found on the blog:
> https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

and was converted from saved_model.pb to handgestures.tflite using the Tensorflow Model Converter:

> https://www.tensorflow.org/lite/models/convert/convert_models#python_api

TENSORFLOW LINK TO handdetection

## Face and Object Detection with Cascade Classifier
To postprocess the images taken in the fotobooth a cascade classifier was used to detect faces in images.
> face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

While giving acceptable results on normal images without any accessories, the results with people wearing big glasses were upgradable.
Therefore a custom cascade classifier was trained to detect the "Big Glasses" in images. The application to train the classifier is:

> https://amin-ahmadi.com/cascade-trainer-gui/




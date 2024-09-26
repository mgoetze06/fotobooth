# Fotobooth
Fotobooth is a python script running on a raspberry pi, controlling a Canon DSLR 1000D through gphoto2. Users can trigger the camera through a footswitch.

<img src="https://github.com/mgoetze06/fotobooth/blob/main/overview.png?raw=true" width="100%">
<img src="https://github.com/mgoetze06/fotobooth/blob/main/1.jpg?raw=true" width="50%">

![CAD](/cad/) Files for the background lights and nerdfacts sign can be found [here](/cad/).

<img src="https://github.com/mgoetze06/fotobooth/blob/main/2.jpg?raw=true" width="50%">
<img src="https://github.com/mgoetze06/fotobooth/blob/main/3.jpg?raw=true" width="25%">

# Webserver
Raspberry PI wlan interface is used to serve as a local Access Point. This wlan interface is used to access a local webserver running Flask and SocketIO. The main script [fotobooth.py](/fotobooth.py) launches the seperate .py script [fotobooth_webserver.py](/webserver/fotobooth_webserver_flask.py) in a subprocess. This is a WIP as the better approach is to use a standard apache-webserver with a WSGI for this. Anyway the locally hosted website is used to interact with the server/fotobooth without the need of a keyboard.

<img src="/webserver/sample_website.png?raw=true" width="80%">

currently implemented:
1) display amount of photos, collages, time
2) display available storage, cpu, cpu temperature
3) set server time
4) shutdown/reboot server
5) download latest image
6) download all images as zip

# Computer Vision and Machine Learning
The fotobooth was part of an educational project at the Leipzig University of Applied Sciences (HTWK Leipzig). The folder [cvml](/cvml/) contains all scripts and notebooks developed during the project. The whole project relies heavily on opencv for python.
>https://pypi.org/project/opencv-python/

The cvml-project consists of two seperate parts:
1) hand detection and gesture recognition (mediapipe + tensorflow lite)
2) face (and object) detection (opencv cascade classifier)

## Mediapipe hand detection
The project uses the mediapipe framework from google to detect hand landmarks on an image. It returns an 1x21x2 Array containing the position of finger joints and finger tips in the image.

> https://google.github.io/mediapipe/solutions/hands.html

<img src="https://github.com/mgoetze06/fotobooth/blob/main/cvml/results/thumbup_mediapipe.png?raw=true" width="30%">

## Tensorflow Gesture Recognition
The landmarks that were detected by mediapipe are processed by a tensorflow lite model to recognize the gestures "thumbs up" and "thumbs down". The model was found on the blog:
> https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

and was converted from saved_model.pb to handgestures.tflite using the Tensorflow Model Converter:

> https://www.tensorflow.org/lite/models/convert/convert_models#python_api

<img src="https://github.com/mgoetze06/fotobooth/blob/main/cvml/results/thumbup_tensorflow.jpg?raw=true" width="30%">

<img src="https://github.com/mgoetze06/fotobooth/blob/main/cvml/results/handgestures.gif?raw=true" width="75%">

## Face and Object Detection with Cascade Classifier
To postprocess the images taken in the fotobooth a cascade classifier was used to detect faces in images.
> face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

While giving acceptable results on normal images without any accessories, the results with people wearing big glasses were upgradable.
Therefore a custom cascade classifier was trained to detect the "Big Glasses" in images. The application to train the classifier is:

> https://amin-ahmadi.com/cascade-trainer-gui/


The following image of big glasses was detected with [big_glasses_classifier_5](/cvml/big_glasses_classifier_5.xml).

> sample resolution 150x150 <br/>
> max_scale = 1.2 <br/>
> min_neigh = 5 <br/>

<img src="https://github.com/mgoetze06/fotobooth/blob/main/cvml/results/big_glasses_detection.png?raw=true" width="75%">





import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# In[223]:


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils


# In[224]:


# Load the gesture recognizer model

model = load_model('hand-gesture-recognition-code\mp_hand_gesture')
#model = load_model('mp_hand_gesture')
# Load class name
f = open('hand-gesture-recognition-code\gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# In[225]:


thumbs_img = cv2.imread('thumbsup.jpg')
#thumbs_img = cv2.imread('thumbsup2.jpg')
#thumbs_img = cv2.imread('thumbsdown.jpg')
h_org,w_org,temp = thumbs_img.shape
print(thumbs_img.shape)
reduce_factor = 6
thumbs_img = cv2.resize(thumbs_img, (int(w_org/reduce_factor), int(h_org/reduce_factor)))
#thumbs_img = cv2.GaussianBlur(thumbs_img, (5,5), 0)
framergb = cv2.cvtColor(thumbs_img, cv2.COLOR_BGR2RGB)
plt.clf()
plt.imshow(framergb)
x , y, c = framergb.shape
print(framergb.shape)


# In[226]:


def calc_landmark_center(landmarks):
    #for simplicity get average of landmarks and use this as center for a hand
    #TODO: problem for more than 2 detected hands
    x = 0
    y = 0
    samples = len(landmarks)
    for lm in landmarks:
        x = x + lm[0]
        y = y + lm[1]
    x = int(x/samples)
    y = int(y/samples)
    return x,y


# In[227]:


# Get hand landmark prediction

#process twice due to better results
counter = 0
while counter < 2:
    result = hands.process(framergb)
    className = ''
    # post process the result
    if result.multi_hand_landmarks:
        landmarks_pred = []
        landmarks_draw = []
        shape = thumbs_img.shape
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks_pred.append([lmx, lmy])
                landmarks_draw.append([lm.x * shape[1],lm.y * shape[0]])
                cv2.circle(thumbs_img,(int((lmx/x)*y), int((lmy/y)*x)),15,(255, 0, 0))              
            mpDraw.draw_landmarks(thumbs_img, handslms, mpHands.HAND_CONNECTIONS)

    counter = counter + 1
# Drawing landmarks on frames
rx,ry = calc_landmark_center(landmarks_draw)
print(rx,ry)
_ = cv2.circle(thumbs_img,(rx,ry),100,(0, 255, 255))


# In[228]:


# Predict gesture in Hand Gesture Recognition project
for i in range(len(landmarks_pred)//21):
    #print(len(landmarks))
    #print(i)
    #print((21+(i*21)))
    prediction = model.predict([landmarks[i*21:(21+(i*21))]]) #problem with predict if more than one hand is detected
    #prediction = model.predict([landmarks_pred[0:21]])
    #--> landmarks sind dann doppelt so lang wie sie f??r eine hand sein m??ssten

    #landmarks vektor muss vorher geteilt werden auf die anzahl detecteter h??nde und dann erst prediction ausgef??hrt

    pred = prediction.flatten()
    classID = np.argmax(pred[0:len(pred)-1]) #ignore smile
    #classID = np.argmax(pred)
    #classID = np.argmax(prediction)
    #print(prediction)
    
    className = classNames[classID]
    print(className)
      # show the prediction on the frame
    cv2.putText(thumbs_img, className, (10, (50+i*50)), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (0,0,255), 2, cv2.LINE_AA)
plt.imshow(thumbs_img)
cv2.imwrite("thumbs.jpg", thumbs_img)


# In[16]:





# 

# In[ ]:





import cv2

#camera = cv2.VideoCapture(cv2.CAP_V4L2)
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
print("Frame default resolution: (" + str(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)#(7680, 4320
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Frame resolution set to: (" + str(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; ")
# Check if the webcam is opened correctly
if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # When everything is done, release the capture
    camera.release()
    cv2.destroyAllWindows()
# runs the gesture recognition program for my CS 639 project
import numpy as np
import cv2
from HandReader import HandReader

# image stream from web cam
cap = cv2.VideoCapture(0)
reader = HandReader()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Set the first webcam image to the base image
    if not reader.baseImageSet():
        reader.setBaseImage(frame)
        continue
        
    # get the mask for the current frame
    mask = reader.createMask(frame)

    # Display the resulting frame
    cv2.imshow('mask', mask)

    # check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

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
    mask = cv2.merge((mask, mask, mask))

    # write text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,600)
    bottomLeftCornerOfText2 = (10,650)
    fontScale              = 1
    fontColor              = (0,255,0)
    lineType               = 2

    cv2.putText(mask, 'Prediction: ok', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.putText(mask, 'Accuracy: 0.52', bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)


    # Display the resulting frame
    cv2.imshow('CS 639 - Predicting Gestures', mask)

    # check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

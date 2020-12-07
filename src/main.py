# runs the gesture recognition program for my CS 639 project
import numpy as np
import cv2
from HandReader import HandReader
from DispTools import applySubImage, applyStatusText

# image stream from web cam
cap = cv2.VideoCapture(0)
reader = HandReader()

# track curr frame
frame_n = 0

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

    # apply mask to sub image slot
    applySubImage(frame, mask)

    # apply status from model
    preds = ['ok', 'c', 'palm', 'none']
    accs = [0.9922, 0.8812, 0.2123, 0.1112]
    applyStatusText(frame, preds, accs, frame_n)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition - v0.3.9', frame) # mask )

    # update values
    frame_n += 1

    # check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

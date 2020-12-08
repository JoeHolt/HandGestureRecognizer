# runs the gesture recognition program for my CS 639 project
import numpy as np
import cv2
from HandReaderV2 import HandReader
from DispTools import applySubImage, applyStatusText
from ModelWrapper import GestureRecognizer
from ComputerInteraction import ComputerInteraction 

# image stream from web cam
cap = cv2.VideoCapture(0)
reader = HandReader()

# load model
model = GestureRecognizer()
computer = ComputerInteraction()

# track curr frame - we only process every n frames for better FPS
process_every = 5 # process every nth frame
pred_class = []
pred_acc = []
frame_n = 0

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Set the first webcam image to the base image
    if not reader.baseImageSet():
        reader.setBaseImage(frame)
        continue

    # get the mask for the current frame
    segment = reader.segmentImage(frame)
    # segment = cv2.merge((segment, segment, segment))

    if frame_n % process_every == 0:
        # update prediction
        pred_class, pred_acc = model.predict(frame)
        computer.add_prediction(pred_class, pred_acc)

    # apply mask to sub image slot
    applySubImage(frame, segment)

    # apply status from model
    applyStatusText(frame, pred_class[:4], pred_acc[:4], frame_n)

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

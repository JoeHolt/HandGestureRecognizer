# runs the gesture recognition program for my CS 639 project
import numpy as np
import cv2
from time import sleep
from HandReaderV3 import HandReader
from DispTools import applySubImage, applyStatusText
from ModelWrapper import GestureRecognizer
from ComputerInteraction import ComputerInteraction 
from copy import copy

# image stream from web cam
cap = cv2.VideoCapture(0)
read_hand_region = (10, 240, 236, 466) # t r b l
reader = HandReader(read_hand_region)

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

    # get the mask for the current frame
    segment = reader.process(copy(frame))
    if segment is not None:
        segment = cv2.merge((segment, segment, segment))

    if frame_n % process_every == 0 and segment is not None and reader.isCalibrated():
        # update prediction
        pred_class, pred_acc = model.predict(segment)
        computer.add_prediction(pred_class, pred_acc)

    # apply mask to sub image slot
    if segment is not None:
        applySubImage(frame, segment, resize=False)
    else:
        applySubImage(frame, frame)

    # apply status from model
    applyStatusText(frame, pred_class[:4], pred_acc[:4], frame_n)

    # draw the ROI rectangle
    left = read_hand_region[3]
    top = read_hand_region[0]
    right = read_hand_region[1]
    bottom = read_hand_region[2]
    color = (0,255,0) if reader.isCalibrated() else (0,0,255)
    cv2.rectangle(frame, (left, top), (right, bottom+100), color, 2)

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

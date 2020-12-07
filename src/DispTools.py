import numpy as np
import cv2


def applySubImage(frame, img, ratio=0.3):
    """
    Draws the passed image to the bottom right square of the displayed frame

    Params:
    - img: img to draw in bottom right
    - frame: frame to draw image of
    - ratio: ratio to resize image to
    """

    # place mask in bottom right corner
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    new_size = (int(frame_width*ratio), int(frame_height*ratio))
    mask_small = cv2.resize(img, new_size) #, interpolation = cv2.INTER_AREA) 

    # apply small image
    frame[frame_height-new_size[1]:frame_height, frame_width-new_size[0]:frame_width] = mask_small   

def applyRectangle(frame, ratio):
    """
    Applies a white rectangle to the bottom left of the
    frame (to later overlay text)
    """

    # place mask in bottom right corner
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    new_size = (int(frame_width*ratio), int(frame_height*ratio))

    # apply small image
    frame[frame_height-new_size[1]:frame_height, 0:new_size[0]] = 255

def applyStatusText(frame, predictions, accuracy, frame_num, ratio=0.3):
    """
    Displays the given predictions and there accuracy on the screen,
    as well as the current frame that is beind processed

    Parameters:
    - frame: frame to apply text to
    - predictions: array of strings of class names being predicted (max 3)
    - accuracy: array of floats representing accuracy of each class prediction
    - frame_num: current number of frame being processed
    """

    # apply background
    applyRectangle(frame, ratio)

    # text 
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.8
    fontColor              = (0,0,0)
    lineType               = 2

    # poitioning
    inset = 20
    start_height = int(frame.shape[0] - frame.shape[0] * ratio + 30)
    curr_height = start_height

    # header text
    header_text = "Predictions (frame {}):".format(frame_num)
    cv2.putText(frame, header_text, (inset, curr_height), font, fontScale, fontColor, lineType)

    # list predictions
    for idx, (pred, acc) in enumerate(zip(predictions, accuracy)):
        curr_height += 40
        disp_text = "{}.) {} (acc {}%)".format(idx + 1, pred, round(acc*100, 2))
        cv2.putText(frame, disp_text, (inset, curr_height), font, fontScale, fontColor, lineType)






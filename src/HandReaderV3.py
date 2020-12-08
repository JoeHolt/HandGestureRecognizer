import numpy as np
import os
import cv2
import imutils
import time
from scipy.ndimage import gaussian_filter
from copy import copy

class HandReader:
    """
    Hand segmenter based off of
    https://gogul.dev/software/hand-gesture-recognition-p1
    """
    
    def __init__(self, roi_region):
        self.bg = None
        self.processedFrames = 0
        self.aWeight = 0.5
        self.top = roi_region[0] # 10
        self.right = roi_region[1] # 350
        self.bottom = roi_region[2] # 225
        self.left = roi_region[3] # 590
        self.n_calibration_frames = 30

    def isCalibrated(self):
        return self.n_calibration_frames < self.processedFrames

    def process(self, frame):
        # resize and flip
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        # clone the frame
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        # smaller region for better results
        roi = frame[self.top:self.bottom, self.right:self.left]
        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # calibrate over first n frames
        thesholded_ret = None
        if self.processedFrames < self.n_calibration_frames:
            self.run_avg(gray, self.aWeight)
        else:
            # segment the hand
            hand = self.segment(gray)
            if hand is not None:
                # get vals
                (thresholded, segmented) = hand
                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (self.right, self.top)], -1, (0, 0, 255))
                thesholded_ret = thresholded

        # increment the number of frames
        self.processedFrames += 1

        return thesholded_ret

    def run_avg(self, image, aWeight):
        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
        else:
            cv2.accumulateWeighted(image, self.bg, aWeight)

    def segment(self, image, threshold=25):

        diff = cv2.absdiff(self.bg.astype("uint8"), image) # find different between averaged bacground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1] # seperate forground and bg
        (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get areas in thresh
        if len(cnts) == 0:
            return
        else:
            segmented = max(cnts, key=cv2.contourArea) # get the biggest segment
            return (thresholded, segmented)


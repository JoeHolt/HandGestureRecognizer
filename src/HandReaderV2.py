import numpy as np
import os
import cv2
from scipy.ndimage import gaussian_filter
from copy import copy

class HandReader:
    """
    A class that creates masks from images as an image preprocessing step
    """
    
    def __init__(self, base_img_path=None):
        """
        Sets up class
        
        Parameters:

        base_img (str): 
        - base image to create masks from
        - camera should not move between taking base image and other images
        """
        if base_img_path is not None:
            self.im_base = self._readImage(base_img_path)
        else:
            self.im_base = None

    def baseImageSet(self):
        return self.im_base is not None

    def setBaseImage(self, img):
        """
        Sets base image to the passed image value
        """
        self.im_base = self._cvt_gray(img)

    def segmentImage(self, fg):
        # gray
        fg_gray = self._cvt_gray(fg)
        # segment
        diff = fg_gray - self.im_base
        # mask
        diff_bg_idxs = (diff < 200)
        # smooth
        smoothed = gaussian_filter(diff_bg_idxs, sigma=0.2)
        smoothed = np.invert(smoothed)
        # apply
        applied = copy(fg)
        applied[smoothed == True] = 0

        return applied
        


    def _cvt_gray(self, img):
        """
        Converts passed image to BW image based on a threshold
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

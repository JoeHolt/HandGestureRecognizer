import numpy as np
import os
import cv2
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
        fg_seg, fg_idxs = self._create_mask(fg)
        return np.invert(fg_seg) 

    def _cvt_gray(self, img):
        """
        Converts passed image to BW image based on a threshold
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _create_mask(self, fg, smooth = True):
        """
        Creates a mask based on a new foreground
        """
        # cvt
        fg_gray = self._cvt_gray(fg)
        # subtract
        mask = cv2.subtract(self.im_base, fg_gray)
        # smooth
        mask_smooth = mask
        if smooth:
            kernel = np.ones((5,5),np.float32)/25
            mask_smooth = cv2.filter2D(mask, -1, kernel)
        # return
        mask_idx = (mask_smooth <= 100)
        return mask_smooth, mask_idx

    def _apply_mask(self, fg, mask):
        new_img = copy(fg)
        new_img[mask] = 0
        return new_img
        

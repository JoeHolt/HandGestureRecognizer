import numpy as np
import os
import cv2

class HandReader:
    """
    A class that creates masks from images as an image preprocessing step
    """
    
    def __init__(self, base_img_path):
        """
        Sets up class
        
        Parameters:

        base_img (str): 
        - base image to create masks from
        - camera should not move between taking base image and other images
        """
        self.im_base = self._readImage(base_img_path)
        
    def _readImage(self, path, bw=True):
        """
        Reads the image at the given location and returns it as an np array
        - Reads in BW by default
        """
        img = cv2.imread(path)

        if bw:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
            return img_bw
        
        return img
        
    def createMask(self, img_path):
        """
        Creates a new mask from the passed image in relation to
        the base image
        
        Parameters:
        img (ndarray):
        - Image object
        
        Returns: Image of same size as input representing mask of difference
        """
        img = self._readImage(img_path)
        assert(img.shape == self.im_base.shape)
        
        mask = cv2.subtract(self.im_base, img)
        
        return mask
        
import numpy as np
import os
import cv2

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
        self.im_base = self._processImage(img)

    def _processImage(self, img):
        """
        Transforms image into BW
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
        return img_bw
        
    def _readImage(self, path, process=True):
        """
        Reads the image at the given location and returns it as an np array
        - Reads in BW by default
        """
        img = cv2.imread(path)

        if process:
            return self._processImage(img) 
        return img
        
    def createMask(self, img):
        """
        Creates a new mask from the passed image in relation to
        the base image
        
        Parameters:
        img (ndarray):
        - Image object
        
        Returns: Image of same size as input representing mask of difference
        """
        assert self.im_base is not None, "Error: im_base not set"

        # read image if we were passed a path
        if isinstance(img, str):
            img = self._readImage(img_path, process=False)

        # process image for our pipeline
        img_proc = self._processImage(img)

        # make sure we have valid image
        assert img_proc.shape == self.im_base.shape, "Invalid shapes: base = {}, new = {}".format(self.im_base.shape, img.shape)
        
        mask = cv2.subtract(self.im_base, img_proc)
        
        return mask
        

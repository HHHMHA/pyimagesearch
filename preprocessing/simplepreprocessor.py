import cv2
import numpy as np


class SimplePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """
        Creates an converter for an image to numpy array resizing it as specified.

        :param width: The desired width of the image
        :param height: The desired height of the image
        :param interpolation: The openCV interpolation method to use
        """
        self.width = width
        self.interpolation = interpolation
        self.height = height

    def preprocess(self, image: any) -> np.ndarray:
        """
        Resize the image as specified and convert it to numpy array.

        :param image: The image to resize
        :return: resized numpy image
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)

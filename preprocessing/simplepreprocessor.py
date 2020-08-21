import cv2
import numpy as np


class SimplePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.interpolation = interpolation
        self.height = height

    def preprocess(self, image: np.ndarray):
        cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)

import imutils
import cv2
from numpy import ndarray


class AspectAwarePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.interpolation = interpolation
        self.height = height
        self.width = width

    def preprocess(self, image: ndarray):
        h, w = image.shape[:2]
        dW, dH = 0, 0

        if w < h:
            image = imutils.resize(image,
                                   width=self.width,
                                   inter=self.interpolation)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image,
                                   height=self.height,
                                   inter=self.interpolation)
            dW = int((image.shape[1] - self.width) / 2.0)

        h, w = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image,
                          (self.width, self.height),
                          interpolation=self.interpolation)

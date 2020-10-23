import numpy as np
from keras_preprocessing.image import img_to_array
from . import data_format


class ImageToArrayPreprocessor:
    def __init__(self, data_format: str = data_format.CHANNELS_LAST):
        """
        Creates an converter for an image to numpy array storing the channel layer in the specified order.

        :param data_format: Order of the channel layer.
        """
        self.data_format = data_format

    def preprocess(self, image: any) -> np.ndarray:
        """
        Converts the image to numpy array with the channels layer at the specified order

        :param image: The image to convert
        :return: numpy array with the channels layer at the specified order
        """
        return img_to_array(image, data_format=self.data_format)

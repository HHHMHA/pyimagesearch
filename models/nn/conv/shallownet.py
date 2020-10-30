from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.activations import relu, softmax
from keras import backend as K

from preprocessing import data_format


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (width, height, depth)

        if K.image_data_format() == data_format.CHANNELS_FIRST:
            input_shape = (depth, width, height)

        model.add(Conv2D(32, (3, 3), activation=relu, input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(classes, activation=softmax))

        return model

import keras
from keras import models
from keras import layers, activations

from preprocessing import data_format


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = models.Sequential()
        input_shape = (height, width, depth)
        chan_dim_index = -1

        if keras.backend.image_data_format() == data_format.CHANNELS_FIRST:
            input_shape = (depth, height, width)
            chan_dim_index = 1  # 0 is batch index

        model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=input_shape))
        model.add(layers.BatchNormalization(axis=chan_dim_index))
        model.add(layers.Conv2D(32, (3, 3), activation=activations.relu))
        model.add(layers.BatchNormalization(axis=chan_dim_index))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
        model.add(layers.BatchNormalization(axis=chan_dim_index))
        model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
        model.add(layers.BatchNormalization(axis=chan_dim_index))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation=activations.relu))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(classes, activation=activations.softmax))

        return model

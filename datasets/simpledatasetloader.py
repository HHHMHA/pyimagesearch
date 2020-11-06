from typing import List, Any

import numpy as np
import cv2
import os
from preprocessing.simplepreprocessor import SimplePreprocessor

# TODO move functions to a util module or someting

# ex: ./dataset/cats/cat1.jpg when splitting by / the last index (-1) will be the
# image name (cat1.jpg) and the one before it (-2) should be the label (cats)
LABEL_DIRECTORY_INDEX = -2


def get_image_label(image_path: str) -> str:
    return image_path.split(os.path.sep)[LABEL_DIRECTORY_INDEX]


def try_print_progress(progress: int, goal: int, verbose: int) -> None:
    """
    Prints progress every 'verbose' (ex: if verbose is 3 it will print the progress at 3, 6, 9, ...)

    :param progress: The progress you want to print
    :param goal: The value progress is going for
    :param verbose: Interval between prints
    """
    if verbose > 0 and progress > 0 and (progress + 1) % verbose == 0:
        print(f"[INFO] processed {progress + 1}/{goal}")


class SimpleDatasetLoader:
    """
    This class assumes the dataset can fit in the memory.
    If the dataset can't fit in the main memory you can't use it
    """

    def __init__(self, preprocessors: List[SimplePreprocessor] = None):
        self.preprocessors = preprocessors if preprocessors is not None else []

    def load(self, image_paths: List[str], verbose=-1) -> (np.ndarray, np.ndarray):
        data: List[np.ndarray] = []
        labels: List[str] = []

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = self.preprocess_image(image)

            label = get_image_label(image_path)

            data.append(image)
            labels.append(label)

            try_print_progress(i, len(image_paths), verbose)

        return np.array(data), np.array(labels)

    def preprocess_image(self, image):
        image_copy = np.copy(image)
        for preprocessor in self.preprocessors:
            image_copy = preprocessor.preprocess(image_copy)
        return image_copy

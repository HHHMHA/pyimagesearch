import argparse

import cv2
import numpy as np
from imutils import paths
from keras.engine.saving import load_model

from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

class_labels = ["cat", "dog"]
print("[INFO] sampling images...")
image_paths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(image_paths), size=(10, ))
image_paths = image_paths[idxs]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    cv2.putText(image, f"Label {class_labels[preds[i]]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(image)
    cv2.waitKey(0)

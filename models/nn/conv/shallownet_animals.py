import argparse

from imutils import paths
from keras import optimizers, losses, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from datasets.simpledatasetloader import SimpleDatasetLoader
from models.nn.conv.shallownet import ShallowNet
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255

labels = LabelBinarizer().fit_transform(labels)
train_X, train_y, test_X, test_y = train_test_split(data, labels, test_size=0.2)

print('[INFO] compiling model...')
model = ShallowNet().build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=losses.categorical_crossentropy, metrics=[metrics.binary_accuracy])

H = model.fit(train_X, train_y, batch_size=32, epochs=100, validation_data=(test_X, test_y))

print('[INFO] evaluating network...')
pred = model.predict(test_X, batch_size=32)
print(classification_report(test_y.argmax(axis=1), pred.argmax(axis=1), target_names=["cat", "dog"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

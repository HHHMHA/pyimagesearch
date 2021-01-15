from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from models.nn.conv.minivggnet import MiniVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

LABEL_PATH_INDEX = -2
WIDTH = 64
HEIGHT = 64

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

print('[INFO] loading dataset')
image_paths = list(paths.list_images(args['dataset']))
class_names = [image_path.split(os.path.sep)[LABEL_PATH_INDEX] for image_path in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aap = AspectAwarePreprocessor(WIDTH, HEIGHT)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader([aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

train_X, test_X, train_y, test_y = train_test_split(data,
                                                    labels,
                                                    test_size=0.25)
label_binarizer = LabelBinarizer()
train_y = label_binarizer.fit_transform(train_y)
test_y = label_binarizer.transform(test_y)

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

print('[INFO] compiling model')
model = MiniVGGNet.build(width=WIDTH, height=HEIGHT, depth=3, classes=len(class_names))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(0.1),
              metrics=['accuracy'])

print('[INFO] training model')
H = model.fit_generator(
    aug.flow(train_X, train_y, batch_size=32),
    validation_data=(test_X, test_y),
    steps_per_epoch=len(train_X) // 32,
    epochs=100,
    verbose=1
)

print('[INFO] evaluating model')
preds = model.predict(test_X, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
                            preds.argmax(axis=1),
                            target_names=class_names))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='validation accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

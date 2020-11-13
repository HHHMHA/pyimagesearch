import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import  classification_report
from .minivggnet import MiniVGGNet
from keras import datasets
from keras import optimizers, losses, metrics
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
(train_X, train_y), (test_X, test_y) = datasets.cifar10.load_data()
train_X = train_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model")
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.RMSprop(lr=0.001),
    metrics=[metrics.categorical_accuracy]
)

H = model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=64, epochs=40, verbose=1)

print("[INFO] evaluating the network...")
pred = model.predict(test_X, batch_size=64)
print(classification_report(test_y.argmax(axis=1), pred.argmax(axis=1), labels=label_names))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
plt.show()

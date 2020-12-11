import imutils
import cv2


def preprocess(image, width, height):
    h, w = image.shape[:2]
    image = imutils.resize(image, width=width) if w > h else imutils.resize(image, height=height)

    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image


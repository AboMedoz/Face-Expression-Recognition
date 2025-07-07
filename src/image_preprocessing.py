import cv2
import numpy as np


def preprocess_img(img, axis, cvt_to_gray):
    if cvt_to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis)
    return img

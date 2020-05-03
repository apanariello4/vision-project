from cv2 import cv2
import numpy as np


def erosion_dilation(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(
        (3, 3), dtype=np.uint8), iterations=1)
    img = cv2.erode(img, np.ones(
        (3, 3), dtype=np.uint8), iterations=1)

    return img


def resize_to_ratio(img, ratio):
    """
    Resize an image according to the given ration
    :param img: Image to be resized
    :param ratio: ratio used to resize the image
    :return: Image resized
    """
    assert ratio > 0, 'ratio_percent must be > 0'
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resize_when_too_big(img, threshold_w_h):
    h = int(img.shape[0])
    w = int(img.shape[1])
    thr_w, thr_h = threshold_w_h
    if h > thr_h or w > thr_h:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        ratio = min(h_ratio, w_ratio)
        img = resize_to_ratio(img, ratio)
    return img


def histogram(img):
    """
    """
    pass


def entropy(hist):
    """
    """
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

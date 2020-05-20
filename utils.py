from cv2 import cv2
import numpy as np


def erosion_dilation(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(
        (4, 4), dtype=np.uint8), iterations=1)
    # img = cv2.erode(img, np.ones(
    #    (3, 3), dtype=np.uint8), iterations=1)

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
    nbin = 255
    color_histogram = []

    for c in range(3):
        histogram = np.zeros((nbin,))
        for row in range(img.shape[1]):
            for col in range(img.shape[2]):
                pixel = img[c, row, col]
                bin = pixel * nbin // 256
                histogram[bin] += 1

        color_histogram = np.concatenate((color_histogram, histogram))
    color_histogram = color_histogram / np.sum(color_histogram)

    return color_histogram


def entropy(hist):
    """
    """
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def crop_image(img, coordinates):
    """
    """
    x, y, w, h = coordinates
    crop_img = img[y:y + h, x:x + w]
    cv2.imshow("crop", crop_img)
    return crop_img


def equalize_luma(img):
    """
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(img)
    Y = cv2.equalizeHist(Y)
    img = cv2.merge((Y, U, V))
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    return img

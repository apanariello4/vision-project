import cv2 as cv2
import numpy as np
from utils import erosion_dilation
from painting_detection import auto_canny
import random


def draw_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


def image_segmentation(img):
    """
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=10)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (7, 7), iterations=2)

    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)  # 11,2 at the beginning

    cv2.imshow("Thresh", gray)
    # gray = erosion_dilation(gray)

    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
        gray, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA)

    return labeled_img, stats


def get_rois(img, gray_img):
    contours = cv2.findContours(
        gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ROI = img[y:y+h, x:x+w]
        cv2.imshow('ROI', ROI)


def convex_hull(img):
    """
    """
    canny_output = auto_canny(img)
    contours, _ = cv2.findContours(
        canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros(
        (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
       # cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    return drawing

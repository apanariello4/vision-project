import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cv2 import cv2
import numpy as np
import random as rng
import utils

def contours(img, adaptive=True):

    blur = cv2.medianBlur(img, 5)
    grayscale = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    if adaptive is False:
        _, thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(
            grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2
        )

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    # opening = cv2.morpholfirst_frameyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def hough_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, np.array([]), 50, 5)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def draw_contours(img, contours, approximate=False):
    # draw in blue the contours that were founded
    cv2.drawContours(img, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)

    if approximate is True:
        # approximate the contour (approx) with 10% tolerance
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
    else:
        x, y, w, h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_corners(img):
    blur = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]


def auto_canny(image, sigma=0.33):
    """
    finds the optimal threshold parameters
    using the median of the image
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def convex_hull(img):
    canny_output = cv2.Canny(img, 150, 250)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    return drawing


def overlap(a,b,elements):
    for i in range(elements):
        print("taci",i)
        l1x = a[i][0]
        l1y = a[i][1]
        r1x = a[i][0] + a[i][2]
        r1y = a[i][1] + a[i][3]

        l2x = b[0]
        l2y = b[1]
        r2x = b[0] + b[2]
        r2y = b[1] + b[3]


        # If one rectangle is on left side of other
        if (l1x >= r2x or l2x >= r1x):
            return False

        # If one rectangle is above other
        if (l1y <= r2y or l2y <= r1y):
            return False

        return True


def dram_multiple_contours(img, contours, max_contours = 10, approximate=False):
    # draw in blue the contours that were founded
    image_entropy = img.copy()
    cv2.drawContours(img, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = sorted(contours,key=cv2.contourArea, reverse=True)

    # draw the biggest contour (c) in green
    overlap_area = np.zeros((max_contours,4))
    for i in range(max_contours):
        x, y, w, h = cv2.boundingRect(c[i])
        utils.crop_image(img,(x,y,w,h))

        entropy = utils.entropy(utils.histogram(utils.crop_image(image_entropy, (x, y, w, h))))
        print(overlap_area)

        if entropy > 0:
            if not overlap(overlap_area,(x,y,w,h),i):

                print(overlap(overlap_area, (x, y, w, h), i))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                print(x, y, w, h)
                overlap_area[i,:] = x, y, w, h
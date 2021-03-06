from cv2 import cv2
import numpy as np
from scipy.spatial import distance as dist


def erosion_dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    return dilation


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


def is_roi_outside_frame(frame_width, frame_height, left, top, right, bottom) -> bool:
    """Check if a painting roi is outside of the frame

    :return True: painting is outside frame
    :return False: painting is not outside frame
    """

    if left <= 0 or top <= 0 or right >= frame_width or bottom >= frame_height:
        print("[ERROR] ROI outside frame. Skipping this detection")
        return True
    return False


def order_corners(corners: list) -> list:
    """Takes a list of corners and orders it clockwise

    :param corners: list of corners
    :type corners: list
    :return: ordered list of corners
    :rtype: list
    """
    a = np.sum(corners, axis=1)
    idx = np.argsort(a).astype(np.int32)
    corners = corners[idx, :]
    # [(0, 0), (x, 0), (0, y), (x, y)]

    # if corners.shape[0] == 4:
    #     upper_right = corners[1]
    #     lower_left = corners[2]

    #     if upper_right[0] < lower_left[0]:
    #         # swaps the two coordinates
    #         corners[1], corners[2] = corners[2], corners[1]

    # sort the points based on their x-coordinates
    xSorted = corners[np.argsort(corners[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    corners = np.array([tl, tr, bl, br], dtype="float32")

    return corners


def remove_points_outside_roi(src_points: list, frame_width: int, frame_height: int) -> list:
    """Removes points that are outside of the frame

    :param src_points: list of coordinates
    :type src_points: list
    :param frame_width: width
    :type frame_width: int
    :param frame_height: height
    :type frame_height: int
    :return: list of coordinates inside frame
    :rtype: list
    """

    return [coordinates for coordinates in src_points if (coordinates[0]
                                                          < frame_width) and (coordinates[1] < frame_height)]


def show_img(window_name, img):
    """
    If there is a matched image, the function shows it on screen
        :param img: image to show on screen
    """
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, img)
    cv2.resizeWindow(window_name, int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.waitKey(2)

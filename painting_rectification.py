import math
import operator
from functools import reduce

from cv2 import cv2
import numpy as np
from scipy.spatial import distance as dist

import utils
from painting_detection import auto_canny
from painting_retrieval import RetrieveClass


class RectifyClass():
    def __init__(self, retrieve):
        self.retrieve = retrieve
        print("[INFO] Ready for painting rectification")
        print("___________________________________")

    def rectify(self, painting_roi):
        print("[INFO] Performing painting rectification")
        try:
            # Check if the painting is in db
            ranked_list, dst_points, src_points = self.retrieve.retrieve(
                painting_roi)
            # print("Painting found in Database")
            if not rectify_with_db(painting_roi, ranked_list, dst_points, src_points):
                return None
        except TypeError:
            # print("Painting not found in Database")
            if not rectify_without_db(painting_roi):
                return None

    def rectify_from_3d(self, painting_roi, coordinates, frame):
        left, top, right, bottom = coordinates
        try:
            # Check if the painting is in db
            ranked_list, dst_points, src_points = self.retrieve.retrieve(
                painting_roi)
            image_3d_warped = rectify_3d_with_db(
                painting_roi, ranked_list, dst_points, src_points)

        except TypeError:
            # print("Painting not found in Database")
            if not rectify_without_db(painting_roi):
                return None


def rectify_without_db(painting_roi) -> bool:
    src_points, bbox = get_corners(painting_roi, draw=True)

    if len(src_points) < 4:
        print("[ERROR] Can't find enough corners")
        return None
    src_points = utils.order_corners(src_points)

    x, y, w, h = bbox
    dst_points = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None:
        print(
            "[ERROR] Homography matrix can't be estimated. Rectification aborted.")
        return None
    painting_roi = cv2.warpPerspective(
        painting_roi, H, (w, h))
    print("[SUCCESS] Warped from corners")
    show_img(painting_roi)
    return True


def rectify_with_db(painting_roi, ranked_list, dst_points, src_points) -> bool:
    best = max(ranked_list, key=ranked_list.get)
    match = cv2.imread(best)

    h_match = int(match.shape[0])
    w_match = int(match.shape[1])

    src_points = np.squeeze(src_points, axis=1).astype(np.float32)
    dst_points = np.squeeze(dst_points, axis=1).astype(np.float32)
    # src_points = np.array(utils.remove_points_outside_roi(
    #    src_points, w_match, h_match))

    if src_points.shape[0] < 4:
        src_points, bbox = get_corners(painting_roi, draw=True)

        if len(src_points) < 4:
            print("[ERROR] Can't find enough corners")
            return None
        src_points = utils.order_corners(src_points)

        # dst_point((x, y), (x+w, y), (x+w, y+h), (x, y+h))

        x, y, w, h = bbox
        dst_points = np.array(
            [(0, 0), (w, 0), (0, h), (w, h)])

        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if H is None:
            print(
                "[ERROR] Homography matrix can't be estimated. Rectification aborted.")
            return None
        painting_roi = cv2.warpPerspective(
            painting_roi, H, (w, h))
        print("[SUCCESS] Warped from corners")
        show_img(painting_roi)
    else:

        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if H is None:
            print(
                "[ERROR] Homography matrix can't be estimated. Rectification aborted.")
            return None
        painting_roi = cv2.warpPerspective(
            painting_roi, H, (w_match, h_match))
        #rectify_from_3d(src_points, dst_points, match, painting_roi)  # ----------------------------------------------------------#
        print("[SUCCESS] Warped from keypoints")
        show_img(painting_roi)
    return True


def get_corners(painting_roi, draw=False):
    gray = cv2.cvtColor(painting_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    erosion = cv2.erode(blur, np.ones((9, 9), np.uint8), iterations=2)
    dilation = cv2.dilate(erosion, np.ones((9, 9), np.uint8), iterations=2)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # edges = auto_canny(thresh)

    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    flood = thresh.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(flood)
    im_out = thresh | im_floodfill_inv

    contours, _ = cv2.findContours(
        im_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    bbox = x, y, w, h

    corners = cv2.goodFeaturesToTrack(
        im_out[y-10:y+h+10, x:x+w+10], 4, 0.01, painting_roi.shape[0] / 3, useHarrisDetector=True)
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    corners_img = painting_roi.copy()

    # draw the biggest contour (c) in green
    cv2.rectangle(corners_img, (x, y), (x + w+10, y + h+10), (0, 255, 0), 2)

    if draw:
        for i, corner in enumerate(corners):
            x_corner, y_corner = corner.ravel()
            cv2.circle(corners_img, (x_corner, y_corner), 3, 255, -1)
            cv2.putText(corners_img, f'{i}', (x_corner+3, y_corner+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(255, 0, 0))
        cv2.imshow("corners", corners_img)

    # for corner in corners:
    #     corner[0] += x
    #     corner[1] += y

    return corners, bbox


def show_img(img):
    """
    If there is a matched image, the function shows it on screen
        :param img: image to show on screen
    """
    cv2.namedWindow("Painting rectification", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Painting rectification", img)
    cv2.resizeWindow("Painting rectification", int(
        img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.waitKey(0)


def rectify_3d_with_db(painting_roi, ranked_list, dst_points, src_points) -> bool:
    best = max(ranked_list, key=ranked_list.get)
    match = cv2.imread(best)

    h_match = int(match.shape[0])
    w_match = int(match.shape[1])

    src_points = np.squeeze(src_points, axis=1).astype(np.float32)
    dst_points = np.squeeze(dst_points, axis=1).astype(np.float32)
    src_points = np.array(utils.remove_points_outside_roi(
        src_points, w_match, h_match))

    if src_points.shape[0] < 4:
        return None
    else:
        H, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)
        if H is None:
            print(
                "[ERROR] Homography matrix can't be estimated. Rectification aborted.")
            return None
        img_dataset_warped = cv2.warpPerspective(
            match, H, (painting_roi.shape[1], painting_roi.shape[0]))

        print("[SUCCESS] Warped from keypoints")

        mask = np.all(img_dataset_warped == [0, 0, 0], axis=-1)
        img_dataset_warped[mask] = painting_roi[mask]
        show_img(img_dataset_warped)
        return img_dataset_warped

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

    def rectify_from_3d(self, painting_roi,coordinates,frame):
        left, top, right, bottom = coordinates
        try:
            # Check if the painting is in db
            ranked_list, dst_points, src_points = self.retrieve.retrieve(
                painting_roi)
            image_3d_warped = rectify_3d_with_db(painting_roi, ranked_list, dst_points, src_points)
            if image_3d_warped is None:
                return None
            else:
                #cambiamento = cv2.add(frame,image_3d_warped)
                #cv2.imshow("Immagine sull'originale",cambiamento)
                frame[top:top+image_3d_warped[0],left:left+image_3d_warped[1],:] = image_3d_warped
                cv2.imshow("frame_warpato",frame)
                return image_3d_warped
        except TypeError:
            # print("Painting not found in Database")
            if not rectify_without_db(painting_roi):
                return None


def rectify_without_db(painting_roi) -> bool:
    return None


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
        src_points = get_corners(painting_roi)

        if len(src_points) < 4:
            print("[ERROR] Can't find enough corners")
            return None
        src_points = utils.order_corners(src_points)
        # dst_point((x, y), (x+w, y), (x+w, y+h), (x, y+h))

        dst_points = np.array(
            [(0, 0), (w_match, 0), (0, h_match), (w_match, h_match)])

        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if H is None:
            print("[ERROR] Homography matrix can't be estimated. Rectification aborted.")
            return None
        painting_roi = cv2.warpPerspective(
            painting_roi, H, (w_match, h_match))
        print("[SUCCESS] Warped from corners")
        show_img(painting_roi)
    else:
        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if H is None:
            print("[ERROR] Homography matrix can't be estimated. Rectification aborted.")
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
    dilation = cv2.dilate(blur, np.ones((3, 3), np.uint8), iterations=1)
    _, thresh = cv2.threshold(
        dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # edges = auto_canny(thresh)

    corners = cv2.goodFeaturesToTrack(
        dilation, 4, 0.01, painting_roi.shape[0] / 3)
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    if draw:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(painting_roi, (x, y), 3, 255, -1)

    return corners


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
            print("[ERROR] Homography matrix can't be estimated. Rectification aborted.")
            return None
        img_dataset_warped = cv2.warpPerspective(match, H, (painting_roi.shape[1], painting_roi.shape[0]))

        print("[SUCCESS] Warped from keypoints")
        show_img(img_dataset_warped)
        return img_dataset_warped


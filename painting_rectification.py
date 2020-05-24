import cv2
import numpy as np
from painting_retrieval import RetrieveClass
from painting_detection import auto_canny
from scipy.spatial import distance as dist
from functools import reduce
import operator
import math


class RectifyClass():
    def __init__(self, retrieve):
        self.retrieve = retrieve
        print("Ready for painting rectification")
        print("___________________________________")

    def detect_corners(self, img):
        """
        Finds corners of the image with Corner Harris
        :param img: img in which to find corners
        :return: img with corners drawn
        """
        blur = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]

    def rectify(self, painting_roi):
        ranked_list, dst_points, src_points = self.retrieve.retrieve(painting_roi)
        best = max(ranked_list, key=ranked_list.get)
        match = cv2.imread(best)
        h_match = int(match.shape[0])
        w_match = int(match.shape[1])
        if src_points.shape[0] < 4:
            src_points = self.get_corners(painting_roi, True)
            ##################### dst_point((x, y), (x+w, y), (x+w, y+h), (x, y+h))
            # dst_points = np.array([(0, 0), (w_match, 0), (w_match, h_match), (0, h_match)])
            dst_points = np.array([(0, 0), (0, w_match), (h_match, 0), (h_match, w_match)])
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            painting_roi = cv2.warpPerspective(painting_roi, H, (h_match, w_match))
        else:
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            painting_roi = cv2.warpPerspective(painting_roi, H, (w_match, h_match))
        self.show_img(painting_roi)

    def get_corners(self, painting_roi, draw=False):
        gray = cv2.cvtColor(painting_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        dilation = cv2.dilate(blur, np.ones((3, 3), np.uint8), iterations=1)
        _, thresh = cv2.threshold(
            dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # edges = auto_canny(thresh)

        corners = cv2.goodFeaturesToTrack(dilation, 4, 0.01, painting_roi.shape[0] / 3)
        corners = np.int0(corners)
        corners = np.squeeze(corners)
        if draw:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(painting_roi, (x, y), 3, 255, -1)
        
        return corners

    def Homography(self, img, prev_img):
        if prev_img is None:
            return

        orb = cv2.ORB_create()
        kpt1, des1 = orb.detectAndCompute(prev_img, None)
        kpt2, des2 = orb.detectAndCompute(img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M

    def show_img(self, img):
        """
        If there is a matched image, the function shows it on screen
            :param img: image to show on screen
        """
        cv2.namedWindow("Painting rectification", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Painting rectification", img)
        cv2.resizeWindow("Painting rectification", int(img.shape[1] / 2), int(img.shape[0] / 2))
        cv2.waitKey(0)


def get_painting_from_roi(cntrs, img):
    """
    It takes the contours of a painting and returns the painting extracted from the given image
        :param cntrs: contours of the image
        :param img: image containing paintings
        :return: painting extracted from the image
    """
    # find the biggest countour (c) by the area
    c = max(cntrs, key=cv2.contourArea)

    rc = cv2.minAreaRect(c)
    box = cv2.boxPoints(rc)
    for p in box:
        pt = (p[0], p[1])
        # print(pt)
        # cv2.circle(frame, pt, 5, (200, 0, 0), 2)

    # approximate the contour (approx) with 10% tolerance
    epsilon = 0.1 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)

    # draw the biggest contour (c) in green
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    extracted_painting = img[y: y + h, x: x + w]
    return extracted_painting

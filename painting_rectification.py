import cv2
import numpy as np
from painting_retrieval import RetrieveClass
from painting_detection import auto_canny


class RectifyClass():
    def __init__(self):
        print("Ready for painting retrieval")
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

    def rectify(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        dilation = cv2.dilate(blur, np.ones((3, 3), np.uint8), iterations=1)
        _, thresh = cv2.threshold(
            dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # edges = auto_canny(thresh)
        
        corners = cv2.goodFeaturesToTrack(dilation, 8, 0.01, frame.shape[0] / 2)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, 255, -1)
        cv2.imshow("Corners", frame)
        cv2.waitKey()


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

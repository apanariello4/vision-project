import os
from cv2 import cv2
import numpy as np
import time
import glob
import pickle


def erosion_dilation(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(
        (5, 5), dtype=np.uint8), iterations=2)
    img = cv2.erode(img, np.ones(
        (3, 3), dtype=np.uint8), iterations=3)

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


################################################################################################
################################# PAINTING RETRIEVAL ###########################################
################################################################################################

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


def get_good_matches(matches, thresh=0.6):
    """
    Computes the best matches of 2 paintings_db that exceed a certain threshold
        :param matches: matches from 2 paintings_db
        :param thresh: threshold used to compute the best matches, defaults to 0.6
        :return: best matches that exceed a certain threshold
    """
    good = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good.append([m])
    return good


def check_match(img):
    """
    It shows an image view that shows the best matches between all the paintings_db from database and the actual frame
        :param img: image to show on screen
    """
    cv2.namedWindow("Checking...", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Checking...", img)
    cv2.resizeWindow("Checking...", int(img.shape[1] / 2), int(img.shape[0] / 2))


def show_match(img):
    """
    If there is a matched image, the function shows it on screen
        :param img: image to show on screen
    """
    if img.size != 0:
        print("Match found!")
        cv2.namedWindow("Match", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Match", img)
        cv2.resizeWindow("Match", int(img.shape[1] / 2), int(img.shape[0] / 2))
    else:
        print("No match...")


def compute_and_write_kp(matcher=cv2.ORB_create()):
    """
    Loads the paintings_db, computes keypoints and descriptors. It also writes to
    files the computed keypoints and descriptors.
        :param matcher: the matcher to use (i.e. ORB), defaults to cv2.ORB_create()
        :return: the loaded paintings_db, the computed keypoints and descriptors
    """
    start = time.time()

    images = {}
    keypoints = {}
    descriptors = {}
    kp_temp = {}
    kp_out = open("keypoints_db", "wb")
    desc_out = open("descriptors_db", "wb")

    for file in glob.glob("paintings_db/*.png"):
        images[file] = cv2.imread(file, cv2.IMREAD_COLOR)
        keypoints[file], descriptors[file] = matcher.detectAndCompute(images[file], None)
        index = []
        for point in keypoints[file]:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id)
            index.append(temp)
        kp_temp[file] = index

    pickle.dump(kp_temp, kp_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(descriptors, desc_out, protocol=pickle.HIGHEST_PROTOCOL)

    kp_out.close()
    desc_out.close()
    end = time.time()

    print("[COMPUTING MODE] Loading time: " + "%.2f" % (end - start) + " seconds\n")
    print("___________________________________")

    return images, keypoints, descriptors


def load_keypoints(compute_and_write=False, matcher=cv2.ORB_create()):
    """
    It loads the keypoints and descriptors from files, if any, otherwise it computes them.
    It also returns the loaded paintings_db.
        :param compute_and_write: if True, the function computes the keypoints and descriptors, then it saves them to files.
        If False, the function loads the keypoints and descriptors from files, if any, defaults to False
        :param matcher: the matcher to use (i.e. ORB), defaults to cv2.ORB_create()
        :return: the loaded paintings_db, keypoints and descriptors
    """
    print("___________________________________")
    if compute_and_write:
        print("[Compute_and_write TRUE, switching to computing mode...]")
        return compute_and_write_kp(matcher=matcher)

    print("\n[Starting reading files...]")

    if os.path.exists('descriptors_db') and os.path.exists('keypoints_db'):
        print("[Files found...]")
        with open('descriptors_db', 'rb') as f1:
            descriptors = pickle.load(f1)
        with open('keypoints_db', 'rb') as f2:
            kp_db = pickle.load(f2)
    else:
        print("[Files not found, passing to computing mode...]")
        return compute_and_write_kp(matcher=matcher)
    start = time.time()
    images = {}
    keypoints = {}

    for file in glob.glob("paintings_db/*.png"):
        images[file] = cv2.imread(file, cv2.IMREAD_COLOR)
        kp = []

        for point in kp_db[file]:
            temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                _octave=point[4], _class_id=point[5])
            kp.append(temp)
        keypoints[file] = kp
    end = time.time()
    print("[LOADING MODE] Loading time: " + "%.2f" % (end - start) + " seconds\n")
    print("Starting painting retrieval")
    print("___________________________________")

    return images, keypoints, descriptors


def test():
    """
    Test for debugging
    """
    images2, keypoints2, descriptors2 = compute_and_write_kp()
    images, keypoints, descriptors = load_keypoints()


def print_ranked_list(dictionary):
    """
    It prints on command line a sorted list of matches number between the actual frame and the paintings_db from database
        :param dictionary: unsorted list of matches number
    """
    ranked_list = {
        k: v for k, v in reversed(sorted(dictionary.items(), key=lambda item: item[1]))
    }
    print(ranked_list)
    print("\n#####################################")

################################################################################################
################################################################################################
################################################################################################

import numpy as np
import cv2
import glob
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import painting_detection
import pickle
import time


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
    Computes the best matches of 2 images that exceed a certain threshold
        :param matches: matches from 2 images
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
    It shows an image view that shows the best matches between all the images from database and the actual frame
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
    Loads the images, computes keypoints and descriptors. It also writes to
    files the computed keypoints and descriptors.
        :param matcher: the matcher to use (i.e. ORB), defaults to cv2.ORB_create()
        :return: the loaded images, the computed keypoints and descriptors
    """
    start = time.time()

    images = {}
    keypoints = {}
    descriptors = {}
    kp_temp = {}
    kp_out = open("keypoints_db.txt", "wb")
    desc_out = open("descriptors_db.txt", "wb")

    for file in glob.glob("images/*.png"):
        images[file] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
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
    It also returns the loaded images.
        :param compute_and_write: if True, the function computes the keypoints and descriptors, then it saves them to files.
        If False, the function loads the keypoints and descriptors from files, if any, defaults to False
        :param matcher: the matcher to use (i.e. ORB), defaults to cv2.ORB_create()
        :return: the loaded images, keypoints and descriptors
    """
    print("___________________________________")
    if compute_and_write:
        print("[Compute_and_write TRUE, switching to computing mode...]")
        return compute_and_write_kp(matcher=matcher)

    print("\n[Starting reading files...]")

    if os.path.exists('descriptors_db.txt') and os.path.exists('keypoints_db.txt'):
        print("[Files found...]")
        with open('descriptors_db.txt', 'rb') as f1:
            descriptors = pickle.load(f1)
        with open('keypoints_db.txt', 'rb') as f2:
            kp_db = pickle.load(f2)
    else:
        print("[Files not found, passing to computing mode...]")
        return compute_and_write_kp(matcher=matcher)
    start = time.time()
    images = {}
    keypoints = {}

    for file in glob.glob("images/*.png"):
        images[file] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
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
    It prints on command line a sorted list of matches number between the actual frame and the images from database
        :param dictionary: unsorted list of matches number
    """
    ranked_list = {
        k: v for k, v in reversed(sorted(dictionary.items(), key=lambda item: item[1]))
    }
    print(ranked_list)
    print("\n#####################################")


def init():
    """
    Initialization function
        :return: all we need
    """
    fm = cv2.ORB_create()
    return fm, cv2.BFMatcher(cv2.NORM_HAMMING), cv2.VideoCapture(
        "videos/VIRB0392.mp4"), 0, load_keypoints(compute_and_write=False, matcher=fm),


brisk, bf, cap, frame_number, (images_db, keypoints_db, descriptors_db) = init()

while cap.isOpened():
    start = time.time()

    matched_collage = np.array([    ])
    good_global = 0
    ret, frame = cap.read()
    images_ranked_list = {}
    contours, _ = painting_detection.contours(frame, adaptive=False)
    frame_number += 1

    print("\n############  FRAME N°" + str(frame_number) + "  ############")
    if len(contours) != 0:

        img_frame = get_painting_from_roi(contours, frame)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

        kp_frame, desc_frame = brisk.detectAndCompute(img_frame, None)

        for file in glob.glob("images/*.png"):

            matches = bf.knnMatch(desc_frame, descriptors_db[file], k=2)

            good = get_good_matches(matches)

            collage = cv2.drawMatchesKnn(
                img_frame,
                kp_frame,
                images_db[file],
                keypoints_db[file],
                good,
                None,
                flags=2,
            )

            images_ranked_list[file] = np.asarray(good).shape[0]
            if int(images_ranked_list[file]) > int(good_global):
                good_global = images_ranked_list[file]
                matched_collage = collage

            check_match(collage)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        end = time.time()
        print("\nTime to search the matched image: " + "%.2f" % (
                end - start) + " seconds\n")
        show_match(matched_collage)
        print_ranked_list(images_ranked_list)

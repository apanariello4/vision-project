import numpy as np
import cv2
import glob
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import painting_detection
import pickle


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    """
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    """

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def contours2(img, adaptive=False):
    # blur = cv2.medianBlur(img, 5)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if adaptive is False:
        _, thresh = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        thresh = cv2.adaptiveThreshold(
            grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2
        )

    return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def get_rectified_painting(cntrs, f):
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
    t_img = f[y: y + h, x: x + w]
    return t_img


def get_good_matches(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])
    return good


def check_match(img):
    cv2.namedWindow("Checking...", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Checking...", img)
    cv2.resizeWindow("Checking...", int(img.shape[1] / 2), int(img.shape[0] / 2))


def show_match(img):
    if matched_collage.size != 0:
        print("Corrispondenza trovata")
        cv2.namedWindow("Match", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Match", img)
        cv2.resizeWindow("Match", int(img.shape[1] / 2), int(img.shape[0] / 2))
    else:
        print("Nessuna corrispondenza")


def compute_and_write_kp(matcher=cv2.ORB_create()):
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

    pickle.dump(kp_temp, kp_out)
    pickle.dump(descriptors, desc_out, protocol=pickle.HIGHEST_PROTOCOL)

    kp_out.close()
    desc_out.close()
    return images, keypoints, descriptors


def load_keypoints(compute_and_write=False, matcher=cv2.ORB_create()):
    if compute_and_write:
        return compute_and_write_kp(matcher=matcher)

    images = {}
    keypoints = {}

    with open('descriptors_db.txt', 'rb') as f1:
        descriptors = pickle.load(f1)

    with open('keypoints_db.txt', 'rb') as f2:
        kp_db = pickle.load(f2)

    for file in glob.glob("images/*.png"):
        images[file] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        kp = []

        for point in kp_db[file]:
            temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                _octave=point[4], _class_id=point[5])
            kp.append(temp)
        keypoints[file] = kp
    return images, keypoints, descriptors


def test():
    images2, keypoints2, descriptors2 = compute_and_write_kp()
    images, keypoints, descriptors = load_keypoints()


def print_ranked_list(dictionary):
    ranked_list = {
        k: v for k, v in reversed(sorted(dictionary.items(), key=lambda item: item[1]))
    }
    print(ranked_list)


def init():
    fm = cv2.ORB_create()
    return fm, cv2.BFMatcher(cv2.NORM_HAMMING), cv2.VideoCapture(
        "videos/VIRB0392.mp4"), 0, load_keypoints(compute_and_write=False, matcher=fm)


brisk, bf, cap, frame_number, (images_db, keypoints_db, descriptors_db) = init()

while cap.isOpened():
    matched_collage = np.array([])
    good_global = 0
    ret, frame = cap.read()
    images_ranked_list = {}
    contours, _ = contours2(frame)
    frame_number += 1

    print("\n############  FRAME N°" + str(frame_number) + "  ############\n")
    if len(contours) != 0:

        img_frame = get_rectified_painting(contours, frame)
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

        show_match(matched_collage)
        print_ranked_list(images_ranked_list)

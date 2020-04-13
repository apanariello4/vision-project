import numpy as np
import cv2
import glob


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


def contours_detection(f):
    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    # (success, saliencyMap) = saliency.computeSaliency(frame)

    ########################
    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    ###############################

    # threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY)[1]

    grayscale = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    _, thresh = cv2.threshold(
        grayscale[:, :, 2], 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    t_contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return t_contours, h


def get_painting_frame_max_area(cntrs, f):
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
    t_img = f[y : y + h, x : x + w]
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
    cv2.resizeWindow("Checking...", int(img3.shape[1] / 2), int(img3.shape[0] / 2))


def show_match(img):
    print("Corrispondenza: " + str(name_painting))
    cv2.namedWindow("Match", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Match", img)
    cv2.resizeWindow("Match", int(img.shape[1] / 2), int(img.shape[0] / 2))


brisk = cv2.ORB_create()
cap = cv2.VideoCapture("videos/VIRB0392.mp4")
good_global = 0
name_painting = None
i = 1

filelist = glob.glob("images/*.png")

while cap.isOpened():
    good_global = 0
    name_painting = None
    found = False
    print("\n############ FRAME N°" + str(i))
    i += 1
    ret, frame = cap.read()

    contours, _ = contours_detection(frame)

    if len(contours) != 0:

        img1 = get_painting_frame_max_area(contours, frame)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        kp1, dest1 = brisk.detectAndCompute(img1, None)

        for fname in filelist:

            img2 = cv2.imread(fname)

            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp2, dest2 = brisk.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            matches = bf.knnMatch(dest1, dest2, k=2)

            # Sort them in the order of their distance.
            # matches = sorted(matches, key=lambda x: x.distance)
            good = get_good_matches(matches)

            # # Draw first 10 matches.
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

            dim_attuale = np.asarray(good).shape[0]
            if int(dim_attuale) > int(good_global):
                good_global = dim_attuale
                name_painting = str(fname)
                match_img = img3

            check_match(img3)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        if name_painting:
            show_match(match_img)
            cv2.waitKey(0)
            # Press Q on keyboard to  exit
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
        else:
            print("Nessuna corrispondenza")

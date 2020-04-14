import cv2


def Contours(frame):

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("akjshda", thresh)
    cv2.waitKey()
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    # opening = cv2.morpholfirst_frameyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def Draw_Contours(img, contours):
    # draw in blue the contours that were founded
    cv2.drawContours(img, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)

    # approximate the contour (approx) with 10% tolerance
    epsilon = 0.1 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)

    # draw the biggest contour (c) in green
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

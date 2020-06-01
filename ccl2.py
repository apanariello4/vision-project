import cv2
import numpy as np
import painting_detection

def draw_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    #cv2.imshow('labeled.png', labeled_img)
    #cv2.waitKey()
    return labeled_img

def erosion_dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    img_erosion = cv2.erode(dilation, kernel, iterations=1)
    return img_erosion

def find (img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def labeling(img):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,3) #11,2 at the beginning
    cv2.imshow("thresh", thresh)
    medianBlur = cv2.medianBlur(thresh, 3)
    cv2.imshow("medianBlur", medianBlur)
    post_proc = erosion_dilation(medianBlur)
    cv2.imshow("post_proc", post_proc)
    erosion = cv2.erode(post_proc, kernel, iterations=1)
    dialtion =  cv2.dilate(post_proc, kernel, iterations=2)
    num_labels, labels_im = cv2.connectedComponentsWithAlgorithm(dialtion,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_GRANA)
    labeled_image = draw_components(labels_im)

    return labeled_image
"""
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2) #11,2 at the beginning

    medianBlur = cv2.medianBlur(thresh, 5)

    img_erosion = cv2.erode(medianBlur, kernel, iterations=1)
    cv2.imshow("img_erosion", img_erosion)
    dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imshow("dilation", dilation)
    img_erosion = cv2.erode(dilation, kernel, iterations=1)
    cv2.imshow("img_erosion_ultimo", img_erosion)


    post_proc = erosion_dilation(medianBlur)
    cv2.imshow("post-proc", post_proc)
    num_labels, labels_im = cv2.connectedComponentsWithAlgorithm(img_erosion,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_GRANA)
    labeled_image = draw_components(labels_im)


    return labeled_image

"""
"""
# 1. compute center
# 2. compute ru
# 3. compute rd
# 4. compute r

#Parameters k_1: distortion parameter, k_min-k_max=[0,1] distortion range
#c_x, c_y: center of distortion (0,0)
#n: number of samples of k_1
#eps: acceptable error on k_1

#Canny

#Apply Distortion for k_j^i

# Compute HT

# Apply Gaussian Filter to HT

# Find the maximum M_j of HT_j

# repeat for i<n
"""
import numpy as np


def HTRDC(edges, range_k, n, epsilon):
    """

    :param edges: edges of image obtained with canny edge detector
    :param range_k: tuple containg the min and max of the 
    distortion parameter
    :param n: number of samples of k_1
    :param epsilon: acceptable error
    :return:
    """

    h, w = edges.shape
    k_min, k_max = range_k
    center_x, center_y = get_center(h, w)
    center = np.array((center_x, center_y))
    points_d = get_points(h, w)


    rd = []
    while (k_max - k_min) > epsilon:
        step = (k_max - k_min) / n
        k_range = np.arange(start=k_min + step, stop= k_max, step=step)
        for k in k_range:
            poitns_u = points_d + (points_d - center) * (k * np.sum(points_d)**2)
            ru = np.sqrt(np.sum(np.square(poitns_u), axis=1))
            rd.append(compute_rd(ru, k))
        rd = np.array(rd)


def compute_rd(ru, k):
    """

    :param ru: sqrt(x_u^2 + y_u^2)
    :param k: distortion coefficient
    :return: rd
    """
    rd_1 = np.cbrt(ru / 2*k + np.sqrt((1 / 3*k)**3 + (ru / 2*k)**2))
    rd_2 = np.cbrt(ru / 2*k - np.sqrt((1 / 3*k)**3 + (ru / 2*k)**2))

    return rd_1 + rd_2


def get_center(h, w)
    """
    get the center of the image

    :param h: height of image
    :param w: width of image
    :return: center of image
    """

    c_x = w // 2
    c_y = h // 2

    return c_x, c_y


def get_points(h, w):
    """
    get the coordinates of all image points in shape (h*w, 2)

    :param h: height of image
    :param w: width of image
    :return: coordinates of all image points
    """

    x, y = np.mgrid[0:h,0:w]
    points = np.vstack((x.ravel(), y.ravel())).T

    return points
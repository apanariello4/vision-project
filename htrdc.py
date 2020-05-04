import numpy as np

from skimage.transform import hough_line


def HTRDC(edges, range_k, n, epsilon):
    """

    :param edges: edges of image obtained with canny edge detector
    :param range_k: tuple containg the min and max of the
    distortion parameter
    :param n: number of samples of k_1
    :param epsilon: acceptable error
    :return:
    """
    max_k = None

    h, w = edges.shape
    k_min, k_max = range_k

    center_x, center_y, coordinates = get_center_and_coordinates(h, w)
    center = np.array((center_x, center_y))
    ru = np.sqrt(np.sum(np.square(coordinates), axis=1))
    theta = np.deg2rad(np.arange(start=0, stop=181, step=1))

    # poitns_u = coordinates + (coordinates - center) * (k * np.sum(coordinates)**2)

    while (k_max - k_min) > epsilon:
        step = (k_max - k_min) / n
        k_range = np.arange(start=k_min + step, stop=k_max, step=step)

        if k_range.size < n:
            k_range = np.hstack((k_range, np.array([k_max], dtype=np.float)))
        rd = []
        for k in k_range:
            rd_k = compute_rd(ru, k)
            rd.append(rd_k)

        rd = np.array(rd, dtype=np.float)
        ru_copy = np.tile(ru, n).reshape(rd.shape)
        rs = rd / ru_copy
        ht_max = np.zeros(rs.shape[0], dtype=np.float)

        for i in np.arange(rs.shape[0]):
            undistorted = compute_undistorted(
                edges, center_y, center_x, rs[i], coordinates)
            acc, _, _ = hough_line(undistorted, theta)
            ht_max[i] = np.max(acc)

        argmax = np.argmax(ht_max)
        max_k = k_range[argmax]
        k_max = np.min([max_k + step, 1.])
        k_min = np.max([0., max_k - step])
        n = np.int(np.round(n * 1.1))

    return max_k


def compute_rd(ru, k):
    """
    Compute rd following formula 6 of the paper.
    :param ru:
    :param k:
    :return:
    """
    x = ru / (2*k)
    y = np.sqrt(np.power((1 / (3*k)), 3) + np.square(x))
    rd_1 = np.cbrt(x + y)
    rd_2 = np.cbrt(x - y)

    return rd_1 + rd_2


def get_center_and_coordinates(h, w):
    """
    get the center of the image

    :param h: height of image
    :param w: width of image
    :return: center of image
    """

    c_x = w // 2
    c_y = h // 2

    # (0, 0) coordinates are in the middle of the image so we go from -c to h-c
    x, y = np.mgrid[-c_y:h - c_y, -c_x:w-c_x]
    points = np.vstack((x.ravel(), y.ravel())).T

    points = np.delete(points, c_y*w+c_x, axis=0)

    return c_x, c_y, points


def compute_undistorted(img, cy, cx, rs, points):
    """
    Compute the 'undistorted' version of the image following formula 7
    :param img:
    :param cy:
    :param cx:
    :param rs:
    :param points:
    :return:
    """
    out = np.zeros_like(img)
    rs = np.vstack((rs, rs)).T
    distorted = np.multiply(points, rs)
    distorted = np.round(distorted).astype(np.int)
    out[points[:, 0] + cy, points[:, 1] +
        cx] = img[distorted[:, 0] + cy, distorted[:, 1] + cx]
    out[cy, cx] = img[cy, cx]
    return out


def undistort(img, k):
    h, w = img.shape[0], img.shape[1]
    cx, cy, points = get_center_and_coordinates(h, w)
    ru = np.sqrt(np.sum(np.square(points), axis=1))
    rd = compute_rd(ru, k)
    r = rd / ru
    return compute_undistorted(img, cy, cx, r, points)

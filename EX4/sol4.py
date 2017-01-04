from scipy import ndimage
from sol4_utils import *
from sol4_add import *
import numpy as np


def get_dervative(im):
    kernel = np.array([[1, 0, -1]])
    dx = ndimage.filters.convolve(im, kernel, mode='reflect')
    dy = ndimage.filters.convolve(im, kernel.T, mode='reflect')
    return dx, dy


def harris_corner_detector(im):
    dx, dy = get_dervative(im)
    kernel_size = 3
    k = 0.04
    dx_square = blur_spatial(np.square(dx), kernel_size)
    dy_square = blur_spatial(np.square(dy), kernel_size)
    dx_dy = blur_spatial(np.multiply(dx, dy), kernel_size)
    # R = det(M) - k*(trace(M))^2
    R = (np.multiply(dx_square, dy_square) - np.square(dx_dy)) - np.multiply(k, np.square(dx_square + dy_square))
    R = non_maximum_suppression(R)
    y, x = np.where(R > 0)
    pos = np.column_stack((x, y))
    return pos

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


def transform_coordinates_level(pos, old_level, new_level):
    # 2^(li−lj)(xli, yli)
    factor = 2 ** (old_level - new_level)
    return pos * factor


def sample_descriptor(im, pos, desc_rad):
    K = 1 + 2 * desc_rad

    desc = np.zeros((K, K, pos.shape[0]))
    # TODO: Get rid of the loop
    for i in range(pos.shape[0]):
        # Ensure the point is at lear 'desc_rad' from the edge
        if pos[i, 0] - desc_rad >= 0 and pos[i, 1] - desc_rad >= 0 and \
                                pos[i, 0] + desc_rad < im.shape[0] and pos[i, 1] + desc_rad < im.shape[1]:
            # TODO: correct implementation?
            x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1, step=1)
            y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1, step=1)
            patch = ndimage.map_coordinates(im, (x_pos, y_pos))
            normalized_patch = patch - np.mean(patch)
            desc[:, :, i] = normalized_patch / np.linalg.norm(normalized_patch)

    return desc


def find_features(pyr):
    # TODO: From where do we get m,n,radius, descriptor radius
    pos_l0 = spread_out_corners(pyr, 7, 7, 7)
    pos_l2 = transform_coordinates_level(pos_l0, 0, 2)
    return sample_descriptor(pyr[2], pos_l2, 3)

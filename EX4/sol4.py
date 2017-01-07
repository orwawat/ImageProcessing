from scipy import ndimage
from sol4_utils import *
from sol4_add import *
import numpy as np
from scipy.signal import convolve2d


def get_dervative(im):
    kernel = np.array([[1, 0, -1]])
    # dx = ndimage.filters.convolve(im, kernel, mode='constant')
    # dy = ndimage.filters.convolve(im, kernel.T, mode='constant')
    dx = convolve2d(im, kernel, mode='same')
    dy = convolve2d(im, kernel.T, mode='same')
    return dx, dy


def harris_corner_detector(im):
    dx, dy = get_dervative(im)
    kernel_size = 3
    k = 0.04
    dx_square = blur_spatial(np.square(dx), kernel_size)
    dy_square = blur_spatial(np.square(dy), kernel_size)
    dx_dy = blur_spatial(np.multiply(dx, dy), kernel_size)
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
        # Ensure the point is at least 'desc_rad' from the edge
        if pos[i, 0] - desc_rad >= 0 and pos[i, 1] - desc_rad >= 0 and \
                                pos[i, 0] + desc_rad < im.shape[0] and pos[i, 1] + desc_rad < im.shape[1]:
            # TODO: correct implementation?
            # deal with non integer points
            x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1, step=1)
            y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1, step=1)
            poss = np.dstack(np.meshgrid(x_pos, y_pos)).reshape(-1, 2)
            patch = ndimage.map_coordinates(im, (poss[:, 0], poss[:, 1]), order=1, prefilter=False)
            # patch = im[pos[i, 0] - desc_rad: pos[i, 0] + desc_rad + 1,
            #         pos[i, 1] - desc_rad: pos[i, 1] + desc_rad + 1].flatten()
            normalized_patch = patch - np.mean(patch)
            norm = np.linalg.norm(normalized_patch)
            if (norm != 0):
                desc[:, :, i] = (normalized_patch / norm).reshape(K, K)
            else:
                desc[:, :, i] = (normalized_patch * np.nan).reshape(K, K)

    return desc


def find_features(pyr):
    # TODO: From where do we get m,n,radius, descriptor radius
    desc_rad = 3
    pos_l0 = spread_out_corners(pyr[0], 7, 7, desc_rad)
    pos_l2 = transform_coordinates_level(pos_l0, 0, 2)
    desc = sample_descriptor(pyr[2], pos_l2, desc_rad)
    # desc = sample_descriptor(pyr[0], pos_l0, desc_rad)
    return pos_l0, desc


def calc_descriptor_score(desc1, desc2):
    '''

    :param desc1: descriptor of points from image 1
    :param desc2: descritors of points from image 2
    :return: 2d array of scores between descroptors. in place (i,j) there is the score between descriptor i from image 1
    and descriptor j from image 2
    '''
    s1 = np.ndarray(shape=(desc1.shape[2], desc2.shape[2]))
    # s2 = np.ndarray(shape=(desc2.shape[2], desc1.shape[2]))
    flatten_desc1 = desc1.reshape(-1, desc1.shape[2]).T
    flatten_desc2 = desc2.reshape(-1, desc2.shape[2])
    return np.dot(flatten_desc1, flatten_desc2)
    # flatten_desc1 = desc1.reshape((desc1.shape[2], -1), order='F')
    # flatten_desc2 = desc2.reshape((desc2.shape[2], -1), order='F')

    # TODO: get rid of the loop
    # for i in range(desc1.shape[2]):
    #     for j in range(desc2.shape[2]):
    #         s1[i, j] = np.dot(flatten_desc1[i], flatten_desc2[j])
    #         s1[i, j] = np.dot(desc1[:, :, i].flatten(), desc2[:, :, i].flatten())
    #         if (s1[i, j] > 1 or s1[i, j] < -1):
    #             print('WTf')

    # return s1


def get_highest_indices(arr):
    '''
    :param arr: a 1D array
    :return: the indices of the two highest values
    '''
    arr = np.argpartition(arr * -1, 2)
    result_args = arr[:2]
    return result_args


def match_features(desc1, desc2, min_score):
    S = calc_descriptor_score(desc1, desc2)
    if (S.max() > 1 or S.min() < -1):
        print('XXXXXX descriptors dot product is not in [-1,1] range XXXXXXXXXX')
    match_ind1 = np.empty(shape=(0,), dtype=int)
    match_ind2 = np.empty(shape=(0,), dtype=int)
    for i in range(desc1.shape[2]):
        # get the top two indices from image 2 that match i (i is an index from inage 1)
        k_matchs = get_highest_indices(S[i, :])
        # Check if i is one of the top two indices of the first index
        if (i in get_highest_indices(S[:, k_matchs[0]])) and S[i, k_matchs[0]] > min_score:
            # found a match
            match_ind1 = np.append(match_ind1, i)
            match_ind2 = np.append(match_ind2, k_matchs[0])
        # Check if i is one of the top two indices of the second index
        if (i in get_highest_indices(S[:, k_matchs[1]])) and S[i, k_matchs[1]] > min_score:
            # found a match
            match_ind1 = np.append(match_ind1, i)
            match_ind2 = np.append(match_ind2, k_matchs[1])

    return match_ind1, match_ind2

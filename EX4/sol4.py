from scipy import ndimage
import sol4_utils
import sol4_add
import numpy as np
from scipy.signal import convolve2d
import random
import matplotlib.pyplot as plt


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
    dx_square = sol4_utils.blur_spatial(np.square(dx), kernel_size)
    dy_square = sol4_utils.blur_spatial(np.square(dy), kernel_size)
    dx_dy = sol4_utils.blur_spatial(np.multiply(dx, dy), kernel_size)
    R = (np.multiply(dx_square, dy_square) - np.square(dx_dy)) - np.multiply(k, np.square(dx_square + dy_square))
    R = sol4_add.non_maximum_suppression(R)
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
        # if pos[i, 0] - desc_rad >= 0 and pos[i, 1] - desc_rad >= 0 and \
        #                         pos[i, 0] + desc_rad < im.shape[0] and pos[i, 1] + desc_rad < im.shape[1]:
        x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1, step=1)
        y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1, step=1)
        poss = np.dstack(np.meshgrid(y_pos, x_pos)).reshape(-1, 2)
        patch = ndimage.map_coordinates(im, (poss[:, 0], poss[:, 1]), order=1, prefilter=False)
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
    pos_l0 = sol4_add.spread_out_corners(pyr[0], 7, 7, 12)
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
    flatten_desc1 = desc1.reshape(-1, desc1.shape[2]).T
    flatten_desc2 = desc2.reshape(-1, desc2.shape[2])
    return np.dot(flatten_desc1, flatten_desc2)


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

    # # TODO: DELETE
    # k = np.zeros(shape=(49, desc2.shape[2]))
    # for i in range(desc2.shape[2]):
    #     k[:, i] = desc2[:, :, i].flatten()
    #
    # if (np.allclose(S[17, :],np.dot(desc1[:, :, 17].flatten(), k))):
    #     print('good')
    # else:
    #     print('bad')
    #
    # # DELETE
    for i in range(desc1.shape[2]):
        # get the top two indices from image 2 that match i (i is an index from image 1)
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


def apply_homography(pos1, H12):
    homorgaph_coordiante = np.ones(pos1.shape[0])
    pos1 = np.column_stack((pos1, homorgaph_coordiante))

    pos2_homograph = np.empty(shape=pos1.shape)
    for j in range(pos1.shape[0]):
        pos2_homograph[j, :] = np.dot(H12, pos1[j, :])

    pos2 = np.column_stack((pos2_homograph[:, 0] / pos2_homograph[:, 2], pos2_homograph[:, 1] / pos2_homograph[:, 2]))
    return pos2


def squared_euclidean_distance(v1, v2):
    return ((v1 - v2) ** 2)


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    max_inliers = []
    for i in range(num_iters):
        random_indices = random.sample(range(0, len(pos1) - 1), 4)
        cur_pos1 = points_from_ind(pos1, random_indices)
        cur_pos2 = points_from_ind(pos2, random_indices)
        H12 = sol4_add.least_squares_homography(cur_pos1, cur_pos2)
        if (H12 is None):
            continue

        P2 = apply_homography(pos1, H12)
        current_error = squared_euclidean_distance(P2, pos2)
        current_inliers = (np.where(current_error < inlier_tol)[0])
        if len(current_inliers) > len(max_inliers):
            max_inliers = current_inliers

    H12 = sol4_add.least_squares_homography(points_from_ind(pos1, max_inliers), points_from_ind(pos2, max_inliers))
    return H12, max_inliers


def points_from_ind(pos, indices):
    return np.take(pos, indices, 0)


def display_matches(im1, im2, pos1, pos2, inliers):
    im = np.hstack((im1, im2))
    shift_amount = im1.shape[1]

    plt.imshow(im, plt.cm.gray)

    # for i in range(35, len(pos1)):
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(im1, plt.cm.gray)
    #     plt.scatter(pos1[i, 0], pos1[i, 1])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(im2, plt.cm.gray)
    #     plt.scatter(pos2[i, 0], pos2[i, 1])
    #     plt.show()


    # plt.scatter(pos1[:, 0], pos1[:, 1])
    # plt.scatter(pos2[:, 0] + shift_amount, pos2[:, 1])
    # TODO: get rid of loop
    for i in range(len(pos1)):
        x = [pos1[i, 0], pos2[i, 0] + shift_amount]
        y = [pos1[i, 1], pos2[i, 1]]
        # x = [pos2[i, 0] + shift_amount]
        # y = [pos2[i, 1]]
        if i in inliers:
            plt.plot(x, y, mfc='r', c='y', lw=.4, ms=10, marker='o')
        else:
            plt.plot(x, y, mfc='r', c='b', lw=.4, ms=10, marker='o')

    plt.show()

from scipy import ndimage
import sol4_utils
import sol4_add
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def get_dervative(im):
    kernel = np.array([[1, 0, -1]])
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
    factor = 2 ** (old_level - new_level)
    return pos * factor


def sample_descriptor(im, pos, desc_rad):
    K = 1 + 2 * desc_rad
    desc = np.zeros((K, K, pos.shape[0]))
    for i in range(pos.shape[0]):
        x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1, step=1)
        y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1, step=1)
        poss = np.meshgrid(y_pos, x_pos)
        patch = ndimage.map_coordinates(im, (poss[0].flatten(), poss[1].flatten()), order=1, prefilter=False)
        normalized_patch = patch - np.mean(patch)
        norm = np.linalg.norm(normalized_patch)
        if (norm != 0):
            desc[:, :, i] = (normalized_patch / norm).reshape(K, K)
        else:
            desc[:, :, i] = (normalized_patch * np.nan).reshape(K, K)
    return desc.astype(np.float32)


def find_features(pyr):
    desc_rad = 3
    pos_l0 = sol4_add.spread_out_corners(pyr[0], 7, 7, desc_rad * 2 ** 2)
    pos_l2 = transform_coordinates_level(pos_l0, 0, 2)
    desc = sample_descriptor(pyr[2], pos_l2, desc_rad)
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
    match_ind1 = np.empty(shape=(0,), dtype=int)
    match_ind2 = np.empty(shape=(0,), dtype=int)

    for i in range(desc1.shape[2]):
        # get the top two indices from image 2 that match i (i is an index from image 1)
        k_matchs = get_highest_indices(S[i, :])
        # Check if i is one of the top two indices of the first index

        if S[i, k_matchs[0]] > min_score and (i in get_highest_indices(S[:, k_matchs[0]])):
            # found a match
            match_ind1 = np.append(match_ind1, i)
            match_ind2 = np.append(match_ind2, k_matchs[0])
        # Check if i is one of the top two indices of the second index
        if S[i, k_matchs[1]] > min_score and (i in get_highest_indices(S[:, k_matchs[1]])):
            # found a match
            match_ind1 = np.append(match_ind1, i)
            match_ind2 = np.append(match_ind2, k_matchs[1])

    return match_ind1, match_ind2


def apply_homography(pos1, H12):
    homorgaph_coordiante = np.ones(pos1.shape[0])
    pos1 = np.column_stack((pos1, homorgaph_coordiante))

    pos2_homograph = np.dot(pos1, H12.T)
    zeroIndcies = np.where(pos2_homograph[:, 2] == 0)[0]
    if (len(zeroIndcies) > 0):
        pos2_homograph[zeroIndcies, 0] = np.inf
        pos2_homograph[zeroIndcies, 1] = np.inf

    # pos2_homograph = np.empty(shape=pos1.shape)
    # for j in range(pos1.shape[0]):
    #     pos2_homograph[j, :] = np.dot(H12, pos1[j, :])
    #     if pos2_homograph[j, 2] == 0:
    #         pos2_homograph[j, 2] = -np.inf

    pos2 = np.column_stack((pos2_homograph[:, 0] / pos2_homograph[:, 2], pos2_homograph[:, 1] / pos2_homograph[:, 2]))
    return pos2


def squared_euclidean_distance(v1, v2):
    dist = np.square(np.linalg.norm(v1 - v2, axis=1))
    if dist is None:
        return -np.inf
    else:
        return dist


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    max_inliers = np.array([])
    ind = np.arange(pos1.shape[0])
    for i in range(num_iters):
        H12 = None
        while H12 is None:
            np.random.shuffle(ind)
            random_indices = ind[:4]
            cur_pos1 = points_from_ind(pos1, random_indices)
            cur_pos2 = points_from_ind(pos2, random_indices)
            H12 = sol4_add.least_squares_homography(cur_pos1, cur_pos2)

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

    inliers_points = [points_from_ind(pos1, inliers), points_from_ind(pos2, inliers)]
    plt.plot([inliers_points[0][:, 0], inliers_points[1][:, 0] + shift_amount],
             [inliers_points[0][:, 1], inliers_points[1][:, 1]], mfc='r', c='y', lw=.4, ms=10, marker='o')

    outlier_indices = np.delete(np.arange(start=0, stop=len(pos1)), inliers)
    outlier_points = [points_from_ind(pos1, outlier_indices), points_from_ind(pos2, outlier_indices)]
    plt.plot([outlier_points[0][:, 0], outlier_points[1][:, 0] + shift_amount],
             [outlier_points[0][:, 1], outlier_points[1][:, 1]], mfc='r', c='b', lw=.4, ms=10, marker='o')
    plt.show()


def normalize_homohraphie_matrix(matrix):
    matrix /= matrix[2, 2]
    return matrix


def accumulate_homographies(H_successive, m):
    H2m = [None] * (len(H_successive) + 1)
    H2m[m] = np.eye(3)

    for i in range(m - 1, -1, -1):
        H2m[i] = np.dot(H2m[i + 1], H_successive[i])
        H2m[i] = normalize_homohraphie_matrix(H2m[i])

    for i in range(m + 1, len(H_successive) + 1):
        H2m[i] = np.dot(H2m[i - 1], np.linalg.inv(H_successive[i - 1]))
        H2m[i] = normalize_homohraphie_matrix(H2m[i])

    return H2m


def calc_centers_and_corners(ims, Hs):
    centers = []
    corners = np.empty(shape=(len(ims), 4, 2))

    for i in range(len(ims)):
        corners[i, 0, :] = [0, 0]
        corners[i, 1, :] = [0, ims[i].shape[0]]
        corners[i, 2, :] = [ims[i].shape[1], 0]
        corners[i, 3, :] = [ims[i].shape[1], ims[i].shape[0]]
        centers.append(
            apply_homography(np.array([ims[i].shape[1] // 2, ims[i].shape[0] // 2]).reshape(1, 2), Hs[i])[0][0])
        corners[i, :, :] = apply_homography(corners[i, :, :], Hs[i])
    return centers, corners


def render_panorama_no_blending(ims, Hs):
    centers, corners = calc_centers_and_corners(ims, Hs)
    max_x = np.max(corners[:, :, 0]).astype(np.int)
    min_x = np.min(corners[:, :, 0]).astype(np.int)
    max_y = np.max(corners[:, :, 1]).astype(np.int)
    min_y = np.min(corners[:, :, 1]).astype(np.int)

    panorama = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.float32)

    next_strip = min_x
    y_vec = np.arange(min_y, max_y)

    for i in range(len(ims)):
        prev_strip = next_strip.astype(np.int)
        if i != len(ims) - 1:
            next_strip = ((centers[i] + centers[i + 1]) // 2).astype(np.int)
        else:
            next_strip = max_x.astype(np.int)

        m_grid = np.meshgrid(np.arange(prev_strip, next_strip), y_vec)
        m_coord = np.column_stack((m_grid[0].flatten(), m_grid[1].flatten()))
        i_points = apply_homography(m_coord, np.linalg.inv(Hs[i]))

        i_grid = [i_points[:, 1].reshape(m_grid[0].shape), i_points[:, 0].reshape(m_grid[0].shape)]

        strip = ndimage.map_coordinates(ims[i], i_grid, order=1, prefilter=False)
        panorama[0: max_y - min_y, prev_strip - min_x: next_strip - min_x] = strip

    return panorama


def get_blending_strip_diff(index, num_images):
    blending_size = 64

    if index == 0:
        return 0, blending_size
    elif index == num_images - 1:
        return -blending_size, 0
    else:
        return -blending_size, blending_size


def render_panorama(ims, Hs):
    centers, corners = calc_centers_and_corners(ims, Hs)
    max_x = np.max(corners[:, :, 0]).astype(np.int)
    min_x = np.min(corners[:, :, 0]).astype(np.int)
    max_y = np.max(corners[:, :, 1]).astype(np.int)
    min_y = np.min(corners[:, :, 1]).astype(np.int)

    if (max_y - min_y + 1) % 8 != 0:
        residual = (max_y - min_y + 1) % 8
        min_y += np.floor(residual / 2).astype(np.int)
        max_y -= np.ceil(residual / 2).astype(np.int)

    panorama = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.float32)

    next_strip = min_x
    y_vec = np.arange(min_y, max_y + 1)

    blending_ims = []

    for i in range(len(ims)):
        prev_strip = next_strip.astype(np.int)
        if i != len(ims) - 1:
            next_strip = ((centers[i] + centers[i + 1]) // 2).astype(np.int)
        else:
            next_strip = max_x.astype(np.int)

        prev_strip_diff, next_strip_diff = get_blending_strip_diff(i, len(ims))
        m_grid = np.meshgrid(np.arange(prev_strip + prev_strip_diff, next_strip + next_strip_diff), y_vec)
        m_coord = np.column_stack((m_grid[0].flatten(), m_grid[1].flatten()))
        i_points = apply_homography(m_coord, np.linalg.inv(Hs[i]))

        i_grid = [i_points[:, 1].reshape(m_grid[0].shape), i_points[:, 0].reshape(m_grid[0].shape)]

        strip = ndimage.map_coordinates(ims[i], i_grid, order=1, prefilter=False)

        if prev_strip_diff != 0:
            blending_ims.append(strip[:, : prev_strip_diff * -2])
        if next_strip_diff != 0:
            blending_ims.append(strip[:, next_strip_diff * -2:])
        if len(blending_ims) >= 2:
            mask = np.column_stack((
                np.ones(shape=(strip.shape[0], -prev_strip_diff)),
                np.zeros(shape=(strip.shape[0], -prev_strip_diff))))
            blend_im = sol4_utils.pyramid_blending(blending_ims[0], blending_ims[1], mask, 4, 7, 7)
            panorama[:, prev_strip - min_x + prev_strip_diff: prev_strip - min_x - prev_strip_diff] = \
                blend_im
            del blending_ims[1]
            del blending_ims[0]

            # multiply by two since we make the strip bigger, and we want to skip part of the panorama
        panorama[:, prev_strip - min_x - prev_strip_diff: next_strip - min_x - next_strip_diff] = \
            strip[:, -(prev_strip_diff * 2): strip.shape[1] - (next_strip_diff * 2)]
    return panorama

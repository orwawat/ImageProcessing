from scipy import ndimage
import sol4_utils
import sol4_add
import numpy as np
from scipy.signal import convolve2d
import random
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
    # TODO: Get rid of the loop
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
    # TODO: Change to -2, need to replace nan to -inf?
    arr = np.argpartition(arr * -1, 2)
    result_args = arr[:2]
    return result_args


def match_features(desc1, desc2, min_score):
    S = calc_descriptor_score(desc1, desc2)
    # S = np.array([[0.99, 0.99, 0.5], [0.99, 0.5, 0.7], [0.5,0.99,0.3]])
    # print(S)
    match_ind1 = np.empty(shape=(0,), dtype=int)
    match_ind2 = np.empty(shape=(0,), dtype=int)

    # TODO: Get rid of the loop
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

    # print(match_ind1)
    # print(match_ind2)
    return match_ind1, match_ind2


def apply_homography(pos1, H12):
    homorgaph_coordiante = np.ones(pos1.shape[0])
    pos1 = np.column_stack((pos1, homorgaph_coordiante))

    pos2_homograph = np.empty(shape=pos1.shape)
    for j in range(pos1.shape[0]):
        pos2_homograph[j, :] = np.dot(H12, pos1[j, :])
        if pos2_homograph[j, 2] == 0:
            pos2_homograph[j, 2] = -np.inf

    pos2 = np.column_stack((pos2_homograph[:, 0] / pos2_homograph[:, 2], pos2_homograph[:, 1] / pos2_homograph[:, 2]))
    return pos2


def squared_euclidean_distance(v1, v2):
    return np.square(np.linalg.norm(v1 - v2, axis=1))


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    max_inliers = []
    for i in range(num_iters):
        random_indices = random.sample(range(0, len(pos1) - 1), 4)
        cur_pos1 = points_from_ind(pos1, random_indices)
        cur_pos2 = points_from_ind(pos2, random_indices)
        H12 = sol4_add.least_squares_homography(cur_pos1, cur_pos2)
        if H12 is None:
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

    inliers_points = [points_from_ind(pos1, inliers), points_from_ind(pos2, inliers)]
    # print(inliers_points)
    # print('\n\n\n')
    plt.plot([inliers_points[0][:, 0], inliers_points[1][:, 0] + shift_amount],
             [inliers_points[0][:, 1], inliers_points[1][:, 1]], mfc='r', c='y', lw=.4, ms=10, marker='o')

    outlier_indices = np.delete(np.arange(start=0, stop=len(pos1)), inliers)
    outlier_points = [points_from_ind(pos1, outlier_indices), points_from_ind(pos2, outlier_indices)]
    # plt.plot([outlier_points[0][:, 0], outlier_points[1][:, 0] + shift_amount],
    #          [outlier_points[0][:, 1], outlier_points[1][:, 1]], mfc='r', c='b', lw=.4, ms=10, marker='o')
    plt.show()


def normalize_homohraphie_matrix(matrix):
    # if (matrix[2, 2] == 0):
    #     matrix[2, 2] == np.inf
    matrix /= matrix[2, 2]
    return matrix


def accumulate_homographies(H_successive, m):
    H2m = [None] * (len(H_successive) + 1)
    H2m[m] = np.eye(3)

    # TODO: Check if it's faster with recursive / one for loop
    for i in range(m - 1, -1, -1):
        H2m[i] = np.dot(H2m[i + 1], H_successive[i])
        H2m[i] = normalize_homohraphie_matrix(H2m[i])

    for i in range(m + 1, len(H_successive) + 1):
        H2m[i] = np.dot(H2m[i - 1], np.linalg.inv(H_successive[i - 1]))
        H2m[i] = normalize_homohraphie_matrix(H2m[i])

    return H2m


def calc_centers_and_corners(ims, Hs):
    centers = []  # np.empty(shape=(1, len(ims)))
    corners = np.empty(shape=(len(ims), 4, 2))

    for i in range(len(ims)):
        corners[i, 0, :] = [0, 0]
        corners[i, 1, :] = [0, ims[i].shape[0]]
        corners[i, 2, :] = [ims[i].shape[1], 0]
        corners[i, 3, :] = [ims[i].shape[1], ims[i].shape[1]]
        centers.append(
            apply_homography(np.array([ims[i].shape[1] // 2, ims[i].shape[0] // 2]).reshape(1, 2), Hs[i])[0][0])
        corners[i, :, :] = apply_homography(corners[i, :, :], Hs[i])
    return centers, corners


def render_panorama(ims, Hs):
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


def switch_grid_cartisen(im):
    return im  # np.column_stack((im[:, 1], im[:, 0]))


def render_panorama_blending(ims, Hs):
    centers, corners = calc_centers_and_corners(ims, Hs)
    max_x = np.max(corners[:, :, 0]).astype(np.int)
    min_x = np.min(corners[:, :, 0]).astype(np.int)
    max_y = np.max(corners[:, :, 1]).astype(np.int)
    min_y = np.min(corners[:, :, 1]).astype(np.int)

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

        if (prev_strip_diff != 0):
            blending_ims.append(strip[:, : prev_strip_diff * -2])
        if (next_strip_diff != 0):
            blending_ims.append(strip[:, next_strip_diff * -2:])
        if (len(blending_ims) >= 2):
            mask = np.column_stack((
                np.ones(shape=(strip.shape[0], -prev_strip_diff)),
                np.zeros(shape=(strip.shape[0], -prev_strip_diff))))
            blend_im = \
                switch_grid_cartisen(sol4_utils.pyramid_blending(switch_grid_cartisen(blending_ims[0]),
                                                                 switch_grid_cartisen(blending_ims[1]), mask, 4, 9, 7))
            panorama[:, prev_strip - min_x + prev_strip_diff: prev_strip - min_x - prev_strip_diff] = \
                blend_im
            del blending_ims[1]
            del blending_ims[0]

        # multiply by two since we make the strip bigger, and we want to skip part of the panorama
        panorama[:, prev_strip - min_x - prev_strip_diff: next_strip - min_x - next_strip_diff] = \
            strip[:, -(prev_strip_diff * 2): strip.shape[1] - (next_strip_diff * 2)]
    return panorama


#################################################################################
def harris_corner_detector1(im):
    """
    :param im:      Grayscale image to find key points inside.
    :return:        An array with shape (N,2) of [x,y] key points locations in im.
    """
    corners = np.dstack(np.nonzero(sol4_add.non_maximum_suppression(get_im_response(im))))[0]
    return corners[:, [1, 0]]


def sample_descriptor1(im, pos, desc_rad):
    """
    :param im:          grayscale image to sample within (already in the 3rd pyramid level).

    :param pos:         An array with shape (N,2) of [x,y] positions to sample descriptors in im
                        Note that the coordinates might be non-integers,
                        therefore mapping coordinates is necessary.

    :param desc_rad:    ”Radius” of descriptors to compute

    :return:            A 3D array with shape (K,K,N) containing the ith descriptor at desc(:,:,i).
                        The per-descriptor dimensions KxK are
                        related to the desc rad argument as follows K = 1+2*desc_rad
    """
    # TODO is there a way to avoid looping the N corner points?
    desc = np.empty((1 + 2 * desc_rad, 1 + 2 * desc_rad, pos.shape[0]), dtype=im.dtype)

    for i in range(len(pos)):
        x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1)
        y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1)
        grid = np.meshgrid(y_pos, x_pos, indexing='ij')
        patch = ndimage.map_coordinates(im, grid, order=1, prefilter=False)

        # normalize the patch, by subtracting the mean value, and divide by the norm (after subtracting).
        # note that the norm might be zero, if a pixel equals the mean value of its patch.
        # in this case we put NaN (what else can we do?)
        normalized_patch = patch - np.mean(patch)
        norm = np.linalg.norm(normalized_patch)
        if norm != 0:
            desc[:, :, i] = normalized_patch / norm
        else:
            desc[:, :, i] = np.nan

    return desc


def find_features1(pyr):
    """
    :param pyr:     Gaussian pyramid of a grayscale image having 3 levels.
    :return:        Tuple of 2 arrays as follows:
                    pos:    An array with shape (N,2) of [x,y] feature location per row
                            found in the (third pyramid level of the) image.
                            These coordinates are provided at the pyramid level pyr[0].
                    desc:   A feature descriptor array with shape (K,K,N).
    """
    DESCRIPTOR_RADIUS = 3

    # 3*2**2 as radius for spread_out_corners, to ensure that the 7x7 patch around
    # each corner point in the 3rd pyramid level will be inside the image.
    # for example the point (4,4) in the image will be (1,1) in the 3rd pyramid level,
    # so we want to assure it won't be taken as feature point because the 7x7 patch around
    # it is not contained in the image.
    pos = sol4_add.spread_out_corners(pyr[0], 7, 7, DESCRIPTOR_RADIUS * 2 ** 2)
    # pos = harris_corner_detector(pyr[0])

    desc = sample_descriptor(pyr[2], pos / 4, DESCRIPTOR_RADIUS)

    return pos, desc


def match_features1(desc1, desc2, min_score):
    """

    :param desc1:       A feature descriptor array with shape (K,K,N1)
    :param desc2:       A feature descriptor array with shape (K,K,N2)
    :param min_score:   Minimal match score between two descriptors required
                        to be regarded as corresponding points

    :return:    Tuple of 2 elements:
                match_ind1: Array with shape (M,) and dtype int of matching indices in desc1
                match_ind2: Array with shape (M,) and dtype int of matching indices in desc2.
    """
    n1, n2 = desc1.shape[2], desc2.shape[2]
    k = desc1.shape[0]

    # Reshape desc1 to (n1, k^2) and desc2 to (k^2, n2).
    # desc1 will hold the flattened descriptors in its rows (each row is a descriptor).
    # desc2 will hold the flattened descriptors in its columns (each column is a descriptor).
    # multiplying these two matrices will give the desired Sj,k matrix (match-score=dot-product
    # between jth descriptor in the first frame and the kth descriptor in second frame).
    desc1 = desc1.reshape((k ** 2, n1)).transpose()
    desc2 = desc2.reshape((k ** 2, n2))

    score_matrix = np.dot(desc1, desc2)  # score_matrix is a matrix of shape (n1, n2)

    # Note that the descriptors could have NaN in them (if norm was zero so the element could not
    # be normalized and therefore been set to NaN).
    # If any descriptor that has NaN in it we consider the match to any other descriptor as -infinity.
    # This is because when we take the second maximum element, they will surely not be taken.
    score_matrix[np.isnan(score_matrix)] = -np.inf

    score_matrix_partitioned_by_rows = np.partition(score_matrix, -2, axis=1)
    score_matrix_partitioned_by_cols = np.partition(score_matrix, -2, axis=0)

    second_max_in_each_row = score_matrix_partitioned_by_rows[:, -2].reshape((n1, 1))
    second_max_in_each_col = score_matrix_partitioned_by_cols[-2, :].reshape((1, n2))

    elements_passed_1st_cond = (score_matrix >= second_max_in_each_row).astype(np.int)
    elements_passed_2nd_cond = (score_matrix >= second_max_in_each_col).astype(np.int)
    elements_passed_3rd_cond = (score_matrix > min_score).astype(np.int)

    elements_passed = elements_passed_1st_cond * elements_passed_2nd_cond * elements_passed_3rd_cond

    return elements_passed.nonzero()


def display_matches1(im1, im2, pos1, pos2, inliers):
    """
    This function display a horizontally concatenated image with matched points marked.

    :param im1:         grayscale image
    :param im2:         grayscale image
    :param pos1:        array with shape (N,2) containing N rows of [x,y] coordinates of matched points in im1
    :param pos2:        array with shape (N,2) containing N rows of [x,y] coordinates of matched points in im2
                        (i.e. the match of the ith coordinate is pos1[i,:] in im1 and pos2[i,:] in im2)
    :param inliers:     an array with shape (S,) of inlier matches
    """
    im = np.hstack((im1, im2))
    im2_shift = im1.shape[1]

    plt.imshow(im, cmap=plt.cm.gray)

    plt.plot([pos1[inliers, 0], pos2[inliers, 0] + im2_shift],
             [pos1[inliers, 1], pos2[inliers, 1]],
             mfc='r', c='y', lw=.4, ms=2, marker='o')

    # outliers = np.delete(np.arange(len(pos1)), inliers)
    #
    # plt.plot([pos1[outliers, 0], pos2[outliers, 0] + im2_shift],
    #          [pos1[outliers, 1], pos2[outliers, 1]],
    #          mfc='r', c='b', lw=.4, ms=2, marker='o')

    plt.show()


def apply_homography1(pos1, H12):
    """
    :param pos1:    An array with shape (N,2) of [x,y] point coordinates
    :param H12:     A 3x3 homography matrix

    :return:        An array with the same shape as pos1 with [x,y] point coordinates
                    in image i+1 obtained from transforming pos1 using H12.
    """
    homogeneous_pos = np.ones((pos1.shape[0], 3), dtype=np.float32)
    homogeneous_pos[:, :-1] = pos1

    homogeneous_pos = np.dot(H12, homogeneous_pos.transpose())

    homogeneous_pos = homogeneous_pos / homogeneous_pos[-1, :]
    homogeneous_pos = homogeneous_pos.transpose()

    return homogeneous_pos[:, :-1]


def ransac_homography1(pos1, pos2, num_iters, inlier_tol):
    """
    :param pos1:        An Array with shape (N,2) containing N rows of [x,y] coordinates of matched points.
    :param pos2:        An Array with shape (N,2) containing N rows of [x,y] coordinates of matched points.
    :param num_iters:   Number of RANSAC iterations to perform.
    :param inlier_tol:  inlier tolerance threshold.

    :return:            H12:        A 3x3 normalized homography matrix.
                        inliers:    An Array with shape (S,) where S is the number of inliers,
                                    containing the indices in pos1/pos2 of the maximal set of
                                    inlier matches found.
    """
    n = pos1.shape[0]
    indices = np.arange(n)
    largest_inliers_set = np.array([])

    for i in range(num_iters):

        homography = None
        while homography is None:
            np.random.shuffle(indices)
            rand_indices = indices[:4]
            homography = sol4_add.least_squares_homography(pos1[rand_indices], pos2[rand_indices])

        transformed_points = apply_homography(pos1, homography)

        dist = np.power(np.linalg.norm(transformed_points - pos2, axis=1), 2)

        inliers_set = np.nonzero(dist < inlier_tol)[0]

        if len(largest_inliers_set) < len(inliers_set):
            largest_inliers_set = inliers_set

    homography = sol4_add.least_squares_homography(pos1[largest_inliers_set], pos2[largest_inliers_set])

    return homography, largest_inliers_set


def accumulate_homographies1(H_successive, m):
    """
    :param H_successive:    A list of M−1 3x3 homography matrices where H successive[i] is a homography
                            that transforms points from coordinate system i to coordinate system i+1.
    :param m:               Index of the coordinate system we would like to accumulate
                            the given homographies towards.

    :return:                A list of M 3x3 homography matrices, where H2m[i] transforms points from
                            coordinate system i to coordinate system m.
    """
    H2m = [np.eye(H_successive[0].shape[0])]

    for i in range(m - 1, -1, -1):
        H = np.dot(H2m[0], H_successive[i])
        H2m.insert(0, H / H[2, 2])

    for i in range(m, len(H_successive)):
        H = np.dot(H2m[-1], np.linalg.inv(H_successive[i]))
        H2m.append(H / H[2, 2])

    return H2m


def render_panorama1(ims, Hs):
    """
    :param ims:     A list of N grayscale images (Python list)
    :param Hs:      A list of N 3x3 homography matrices. Hs[i] is a homography that transforms
                    points from the coordinate system of ims[i] to the coordinate system of the
                    panorama. (Python list)

    :return:        A grayscale panorama image composed of vertical strips, backwarped using
                    homographies from Hs, one from every image in ims.
    """
    x_min, x_max, y_min, y_max = get_panorama_corners_coordinates(ims, Hs)
    boundaries = get_boundaries(ims, Hs, x_min, x_max)
    im_panorama = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

    for i in range(len(ims)):
        mesh_grid_in_panorama = np.meshgrid(np.arange(boundaries[i], boundaries[i + 1]),
                                            np.arange(y_min, y_max))

        strip_shape = mesh_grid_in_panorama[0].shape

        x_coord_in_panorama = mesh_grid_in_panorama[0].flatten()
        y_coord_in_panorama = mesh_grid_in_panorama[1].flatten()

        coordinates_in_panorama = np.empty((x_coord_in_panorama.size, 2))
        coordinates_in_panorama[:, 0] = x_coord_in_panorama
        coordinates_in_panorama[:, 1] = y_coord_in_panorama

        transformed_points = apply_homography(coordinates_in_panorama, np.linalg.inv(Hs[i]))

        x_coord_in_orig_im = transformed_points[:, 0]
        y_coord_in_orig_im = transformed_points[:, 1]

        mesh_grid_in_orig_im = [y_coord_in_orig_im.reshape(strip_shape),
                                x_coord_in_orig_im.reshape(strip_shape)]

        strip = ndimage.map_coordinates(ims[i], mesh_grid_in_orig_im, order=1, prefilter=False)
        im_panorama[0: y_max - y_min, boundaries[i] - x_min: boundaries[i + 1] - x_min] = strip

    return im_panorama


def transform_corners(ims, Hs):
    """
    :param ims:     A list of grayscale images (Python list)
    :param Hs:      A list of 3x3 homography matrices. Hs[i] is a homography that transforms
                    points from the coordinate system of ims[i] to the coordinate system of the
                    panorama. (Python list)

    :return:        a list (Python list) of corners coordinates of each image ims[i] transformed by the
                    corresponding homography Hs[i] to corners of ims[i] in the coordinate system
                    of the panorama.
                    this list if has the same number of elements as the number of images in 'ims',
                    and each element is an array containing 4 points (each is array of 2 numbers).
    """
    corners_transformed = []

    for i in range(len(ims)):
        rows, cols = ims[i].shape

        # add corners of each image to corner_orig.
        # note that we are working with x-y coordinates, so x is the columns axis and y is the rows axis.
        # [0, 0] is top-left, [cols, 0] is top-right,
        # [cols, rows] is bottom-right, [0, rows] is bottom-left
        corner_orig = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]])

        # print("corner_orig is:\n", corner_orig)
        # print("Hs[", i, "] is:\n", Hs[i])
        # print("apply_homography(corner_orig, Hs[", i, "]) is:\n", apply_homography(corner_orig, Hs[i]))

        corners_transformed.append(apply_homography(corner_orig, Hs[i]))

    return corners_transformed


def get_panorama_corners_coordinates(ims, Hs):
    """
    :param ims:     A list of grayscale images (Python list)
    :param Hs:      A list of 3x3 homography matrices. Hs[i] is a homography that transforms
                    points from the coordinate system of ims[i] to the coordinate system of the
                    panorama. (Python list)

    :return:        a 4-tuple of corners of the panorama - x_min, x_max, y_min, y_max
    """
    transformed_corners = np.array(transform_corners(ims, Hs))  # shape is (N, 4, 2)

    x_min = np.amin(transformed_corners[:, :, 0]).astype(np.int)
    x_max = np.amax(transformed_corners[:, :, 0]).astype(np.int)
    y_min = np.amin(transformed_corners[:, :, 1]).astype(np.int)
    y_max = np.amax(transformed_corners[:, :, 1]).astype(np.int)

    return x_min, x_max, y_min, y_max


def transform_centers(ims, Hs):
    """
    :param ims:     A list of grayscale images (Python list)
    :param Hs:      A list of 3x3 homography matrices. Hs[i] is a homography that transforms
                    points from the coordinate system of ims[i] to the coordinate system of the
                    panorama. (Python list)

    :return:        a list (Python list) of centers coordinates of each image ims[i] transformed by the
                    corresponding homography Hs[i] to centers of ims[i] in the coordinate system
                    of the panorama.
                    this list if has the same number of elements as the number of images in 'ims',
                    and each element is an array containing 1 points (which is an array of 2 numbers).
    """
    centers_transformed = []

    for i in range(len(ims)):
        rows, cols = ims[i].shape

        # add the center of each image to center_orig.
        # note that we are working with x-y coordinates, so x is the columns axis and y is the rows axis.
        center_orig = np.array([[cols // 2, rows // 2]])

        centers_transformed.append(apply_homography(center_orig, Hs[i]))

    return centers_transformed


def get_boundaries(ims, Hs, x_min, x_max):
    """
    :param ims:                     A list of N grayscale images (Python list)
    :param Hs:                      A list of N 3x3 homography matrices. Hs[i] is a homography that transforms
                                    points from the coordinate system of ims[i] to the coordinate system of the
                                    panorama. (Python list)
    :param x_min:                   x_min coordinate of the panorama coordinates system.
    :param x_max:                   x_max coordinate of the panorama coordinates system.

    :return:                        array containing x-boundaries of images in panorama.
                                    each original image i will be mapped to the coordinates
                                    [y_min, y_max] = full Y range in the y-axis (rows)
                                    [boundaries[i], boundaries[i+1] in the x-axis (columns)
    """

    # transform_centers(ims, Hs) is of shape (N, 1, 2), and we only need the x-values
    transformed_centers_x = np.array(transform_centers(ims, Hs))[:, 0, 0]

    boundaries = np.empty((transformed_centers_x.size - 1,), dtype=np.int)
    for i in range(transformed_centers_x.size - 1):
        boundaries[i] = (transformed_centers_x[i] + transformed_centers_x[i + 1]) // 2

    boundaries = np.insert(boundaries, 0, x_min)
    boundaries = np.append(boundaries, x_max)

    return boundaries


def der_dx(im):
    """
    :param im:  grayscale image to derive
    :return:    the x derivative image, using convolution with [1 0 -1]
    """
    filter_vec = np.array([[1, 0, -1]])
    return convolve2d(im, filter_vec, mode='same')


def der_dy(im):
    """
    :param im:  grayscale image to derive
    :return:    the y derivative image, using convolution with [1 0 -1]^T
    """
    filter_vec = np.array([[1], [0], [-1]])
    return convolve2d(im, filter_vec, mode='same')


def get_im_hessian(im):
    """
    For each pixel in the image, compute its hessian matrix.
    The hessian matrix is 2x2, therefore the im_hessian matrix will be of shape
    (rows, cols, 2, 2) because for each pixel there is a 2x2 matrix
    :param im:  input image
    :return:    hessian matrices image
    """
    BLUR_KERNEL_SIZE = 3

    im_dx = der_dx(im)
    im_dy = der_dy(im)

    im_dx_sqrd = np.power(im_dx, 2)
    im_dy_sqrd = np.power(im_dy, 2)
    dx_mult_dy = np.multiply(im_dx, im_dy)

    im_dx_sqrd_blurred = sol4_utils.blur_spatial(im_dx_sqrd, BLUR_KERNEL_SIZE)
    im_dy_sqrd_blurred = sol4_utils.blur_spatial(im_dy_sqrd, BLUR_KERNEL_SIZE)
    dx_mult_dy_blurred = sol4_utils.blur_spatial(dx_mult_dy, BLUR_KERNEL_SIZE)

    return np.rollaxis(np.rollaxis(np.array([[im_dx_sqrd_blurred, dx_mult_dy_blurred],
                                             [dx_mult_dy_blurred, im_dy_sqrd_blurred]],
                                            dtype=np.float32), 2), 3, 1)


def get_im_response(im):
    """
    :param im:  input image
    :return:    response image, which for each pixel is det(Hessian) - RESPONSE_COEFFICIENT * (trace(Hessian)) ^ 2
    """
    RESPONSE_COEFFICIENT = 0.04

    im_hessian = get_im_hessian(im)
    im_response = \
        np.linalg.det(im_hessian) - RESPONSE_COEFFICIENT * np.power(np.trace(im_hessian, axis1=2, axis2=3), 2)

    return im_response

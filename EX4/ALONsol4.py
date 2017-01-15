from scipy import ndimage
from sol4_utils import *
from sol4_add import *
import scipy.signal as signal
import matplotlib.pyplot as plt

def der_dx(im):
    """
    :param im:  grayscale image to derive
    :return:    the x derivative image, using convolution with [1 0 -1]
    """
    filter_vec = np.array([[1, 0, -1]])
    return signal.convolve2d(im, filter_vec, mode='same')


def der_dy(im):
    """
    :param im:  grayscale image to derive
    :return:    the y derivative image, using convolution with [1 0 -1]^T
    """
    filter_vec = np.array([[1], [0], [-1]])
    return signal.convolve2d(im, filter_vec, mode='same')


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

    im_dx_sqrd_blurred = blur_spatial(im_dx_sqrd, BLUR_KERNEL_SIZE)
    im_dy_sqrd_blurred = blur_spatial(im_dy_sqrd, BLUR_KERNEL_SIZE)
    dx_mult_dy_blurred = blur_spatial(dx_mult_dy, BLUR_KERNEL_SIZE)

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


def harris_corner_detector(im):
    """
    :param im:      Grayscale image to find key points inside.
    :return:        An array with shape (N,2) of [x,y] key points locations in im.
    """
    corners = np.dstack(np.nonzero(non_maximum_suppression(get_im_response(im))))[0]
    return corners[:, [1, 0]]


def sample_descriptor(im, pos, desc_rad):
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
    desc = np.empty((1 + 2 * desc_rad, 1 + 2 * desc_rad, pos.shape[0]), dtype=im.dtype)

    for i in range(len(pos)):
        x_pos = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1)
        y_pos = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1)
        grid = np.meshgrid(y_pos, x_pos, indexing='ij')
        patch = ndimage.map_coordinates(im, grid, order=1, prefilter=False)

        # normalize the patch, y subtracting the mean value, and divide by the norm (after subtracting).
        # note that the norm might be zero, if a pixel equals the mean value of its patch.
        # in this case we put NaN (what else can we do?)
        normalized_patch = patch - np.mean(patch)
        norm = np.linalg.norm(normalized_patch)
        if norm != 0:
            desc[:, :, i] = normalized_patch / norm
        else:
            desc[:, :, i] = np.nan

    return desc


def find_features(pyr):
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
    pos = spread_out_corners(pyr[0], 7, 7, DESCRIPTOR_RADIUS * 2 ** 2)
    desc = sample_descriptor(pyr[2], pos / 4, DESCRIPTOR_RADIUS)

    return pos, desc


def match_features(desc1, desc2, min_score):
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


def apply_homography(pos1, H12):
    """
    :param pos1:    An array with shape (N,2) of [x,y] point coordinates
    :param H12:     A 3x3 homography matrix

    :return:        An array with the same shape as pos1 with [x,y] point coordinates
                    in image i+1 obtained from transforming pos1 using H12.
    """
    homogeneous_pos = np.ones((pos1.shape[0], 3), dtype=np.float32)
    homogeneous_pos[:, :-1] = pos1

    homogeneous_pos = np.dot(H12, homogeneous_pos.transpose())

    # TODO deal with division by zero - put infinity instead
    homogeneous_pos = homogeneous_pos / homogeneous_pos[-1, :]
    homogeneous_pos = homogeneous_pos.transpose()

    return homogeneous_pos[:, :-1]


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
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
            homography = least_squares_homography(pos1[rand_indices], pos2[rand_indices])

        transformed_points = apply_homography(pos1, homography)

        dist = np.power(np.linalg.norm(transformed_points - pos2, axis=1), 2)

        inliers_set = np.nonzero(dist < inlier_tol)[0]

        if len(largest_inliers_set) < len(inliers_set):
            largest_inliers_set = inliers_set

    homography = least_squares_homography(pos1[largest_inliers_set], pos2[largest_inliers_set])

    return homography, largest_inliers_set


def display_matches(im1, im2, pos1, pos2, inliers):
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

    outliers = np.delete(np.arange(len(pos1)), inliers)

    plt.plot([pos1[outliers, 0], pos2[outliers, 0] + im2_shift],
             [pos1[outliers, 1], pos2[outliers, 1]],
             mfc='r', c='b', lw=.4, ms=2, marker='o')

    plt.show()


def accumulate_homographies(H_successive, m):
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
        corner_orig = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]]).astype(np.int)
        corners_transformed.append(apply_homography(corner_orig, Hs[i]).astype(np.int))

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

    # we want the panorama height to be a multiple of 8,
    # so we can use pyramid blending with 4 pyramid levels.
    if (y_max - y_min + 1) % 8 != 0:
        residual = (y_max - y_min + 1) % 8
        y_min += np.floor(residual / 2).astype(np.int)
        y_max -= np.ceil(residual / 2).astype(np.int)

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
    transformed_centers = np.array(transform_centers(ims, Hs))[:, 0, 0]
    boundaries = np.empty((transformed_centers.size - 1,), dtype=np.int)

    for i in range(transformed_centers.size - 1):
        boundaries[i] = (transformed_centers[i] + transformed_centers[i + 1]) // 2

    boundaries = np.insert(boundaries, 0, x_min)
    boundaries = np.append(boundaries, x_max)

    return boundaries


def render_panorama(ims, Hs):
    """
    :param ims:     A list of N grayscale images (Python list)
    :param Hs:      A list of N 3x3 homography matrices. Hs[i] is a homography that transforms
                    points from the coordinate system of ims[i] to the coordinate system of the
                    panorama. (Python list)

    :return:        A grayscale panorama image composed of vertical strips, backwarped using
                    homographies from Hs, one from every image in ims.
    """
    transformed_corners = np.array(transform_corners(ims, Hs))  # shape is (N, 4, 2)
    x_min, x_max, y_min, y_max = get_panorama_corners_coordinates(ims, Hs)

    boundaries = get_boundaries(ims, Hs, x_min, x_max)
    im_panorama = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)
    im_panorama_temp = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

    for i in range(len(ims)):
        leftmost_x = np.ceil(np.amin((transformed_corners[i, 0, 0], transformed_corners[i, 3, 0]))).astype(np.int)
        rightmost_x = np.floor(np.amax((transformed_corners[i, 1, 0], transformed_corners[i, 2, 0]))).astype(np.int)

        mesh_grid_in_panorama = np.meshgrid(np.arange(leftmost_x, rightmost_x), np.arange(y_min, y_max + 1))

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

        # from the second frame onwards, blend the overlapping area of the current image with the previous image
        if i > 0:
            rightmost_x_blending = rightmost_x
            leftmost_x_blending = leftmost_x

            # if for some reason the rightmost and leftmost are not between two sides of boundaries[i],
            # set them artificially to +- 16
            if boundaries[i] - leftmost_x_blending < 0:
                leftmost_x_blending = boundaries[i] - 16
            if rightmost_x_blending - boundaries[i] < 0:
                rightmost_x_blending = boundaries[i] + 16

            # we want the blending width to be a multiple of 8,
            # so we can use pyramid blending with 4 pyramid levels.
            if (rightmost_x - leftmost_x) % 8 != 0:
                residual = (rightmost_x - leftmost_x) % 8
                leftmost_x_blending += np.floor(residual / 2).astype(np.int)
                rightmost_x_blending -= np.ceil(residual / 2).astype(np.int)

            first_im_overlap = im_panorama[:, leftmost_x_blending - x_min: rightmost_x_blending - x_min].copy()
            im_panorama_temp[:, leftmost_x - x_min: rightmost_x - x_min] = strip
            second_im_overlap = im_panorama_temp[:, leftmost_x_blending - x_min: rightmost_x_blending - x_min].copy()

            first_im_mask = np.ones((y_max - y_min + 1, boundaries[i] - leftmost_x_blending))
            second_im_mask = np.zeros((y_max - y_min + 1, rightmost_x_blending - boundaries[i]))

            mask = np.hstack((first_im_mask, second_im_mask)).astype(np.bool)

            # blending parameters
            max_levels = 4
            filter_size_im = 13
            filter_size_mask = 13
            border_width = 32

            blended_overlap_area = pyramid_blending(first_im_overlap, second_im_overlap, mask,
                                                    max_levels, filter_size_im, filter_size_mask)

            im_panorama[:, leftmost_x_blending - x_min + border_width:rightmost_x_blending - x_min] = \
                blended_overlap_area[:, border_width:]
        else:
            im_panorama[:, leftmost_x - x_min: rightmost_x - x_min] = strip

    return im_panorama

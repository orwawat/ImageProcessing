import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray as rgb2gray


def read_image(filename, representation):
    """
    The function reads an image using the given filename.

    :param filename: string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining
                            if the output should be either a grayscale
                            image (1) or an RGB image (2)
    :return: the image
    """

    # change im to be float32 in range [0,1]
    im = imread(filename).astype(np.float32) / 255.0

    if representation == 1:
        im = rgb2gray(im)

    return im.astype(np.float32)


def gaussian_1d(kernel_size):
    """
    Generate a 1D gaussian of shape (kernel_size).
    In order to allow big values, it uses data-type unsigned int 64-bit.
    However, using large values (about 30+) causes overflow and not supported.

    :param kernel_size:     The size of the kernel to filter with.
                            Must be an odd number.
    :return:                The gaussian - a 1-dimensional NumPy array
    """
    base = np.array([1, 1], dtype=np.uint64)
    gaussian = base.copy()

    for i in range(kernel_size - 2):
        gaussian = signal.convolve(gaussian, base)

    return gaussian


def gaussian_2d(kernel_size):
    """
    Generate a 2D gaussian of shape (kernel_size, kernel_size).
    In order to allow big values, it uses data-type unsigned int 64-bit.
    However, using large values (about 30+) causes overflow and not supported.

    :param kernel_size:     The size of the kernel to filter with.
                            Must be an odd number.
    :return:                The gaussian - a 2-dimensional matrix
    """

    gaus1d = gaussian_1d(kernel_size)

    gaus2d = np.outer(gaus1d, gaus1d)
    gaus2d = gaus2d / np.sum(gaus2d)

    return gaus2d


def blur_spatial(im, kernel_size):
    """
    Performs image blurring using 2D convolution between the image f
    and a gaussian kernel

    :param im:          The image to blur
    :param kernel_size: The gaussian kernel size (mus be an odd integer)
    :return:            The blurred image
    """

    return signal.convolve2d(im, gaussian_2d(kernel_size), mode='same')


def stretch_im(im):
    """
    :param im:  image to stretch
    :return:    image with stretched values between 0 and 1
    """
    min_value, max_value = np.amin(im), np.amax(im)
    im_stretched = ((im - min_value) / (max_value - min_value))

    return im_stretched.astype(np.float32)


def stretch_pyr(pyr):
    """
    :param pyr: pyramid of images to stretch
    :return:    pyramid of stretched images with values between 0 and 1
    """
    pyr_stretched = []

    for im in pyr:
        pyr_stretched.append(stretch_im(im))

    return pyr_stretched


def reduce(im, filter_vec):
    """
    Reduces an image - blurring using 2D convolution between the image
    and a gaussian kernel (of the given kernel_size. The boundaries are
    handled symmetrically).
    Afterwards, sub-sample by taking its even indexes (assuming zero-index)

    :param im:          The image to reduce.
    :param filter_vec:  The vector to filter with.
    :return:            The reduced image.
    """
    filter_col_vec = filter_vec.reshape((filter_vec.size, 1))
    filter_row_vec = filter_col_vec.transpose()

    new_im = signal.convolve2d(im, filter_col_vec, mode='same')
    new_im = signal.convolve2d(new_im, filter_row_vec, mode='same')

    return new_im[::2, ::2].astype(np.float32)


def expand(im, filter_vec):
    """
    Expands an image - Padding odd indices with zeros and blurring with 2*gaussian.
    The boundaries are handled symmetrically).

    :param im:          The image to expand.
    :param filter_vec:  The vector to filter with.

    :return:            The expanded image.
    """
    im_expanded = np.zeros(2 * np.array(im.shape), dtype=im.dtype)
    im_expanded[::2, ::2] = im

    filter_col_vec = 2 * filter_vec.reshape((filter_vec.size, 1))
    filter_row_vec = filter_col_vec.transpose()

    im_expanded = signal.convolve2d(im_expanded, filter_col_vec, mode='same')
    im_expanded = signal.convolve2d(im_expanded, filter_row_vec, mode='same')

    return im_expanded.astype(np.float32)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Construct a Gaussian pyramid pyramid of a given image

    :param im:              a grayscale image with double values in [0, 1]
                            (e.g. the output of ex1s read_image with the
                            representation set to 1).
    :param max_levels:      the maximal number of levels in the resulting pyramid.
    :param filter_size:     the size of the Gaussian filter (an odd scalar that
                            represents a squared filter) to be used
                            in constructing the pyramid filter
    :return:                a Gaussian pyramid pyramid of a given image
    """
    pyr = [im]
    filter_vec = gaussian_1d(filter_size).reshape((1, filter_size))
    filter_vec = filter_vec / np.sum(filter_vec)

    for i in range(1, max_levels):
        im_reduced = reduce(pyr[-1], filter_vec)
        pyr.append(im_reduced)

        # if the current reduced image has minimum dimension less that 32 it means
        # that in the next iteration the dimension will be less then 16, so break.
        if min(im_reduced.shape[0], im_reduced.shape[1]) < 32:
            break

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Construct a Laplacian pyramid pyramid of a given image

    :param im:              a grayscale image with double values in [0, 1]
                            (e.g. the output of ex1s read_image with the
                            representation set to 1).
    :param max_levels:      the maximal number of levels in the resulting pyramid.
    :param filter_size:     the size of the Gaussian filter (an odd scalar that
                            represents a squared filter) to be used
                            in constructing the pyramid filter
    :return:                a Laplacian pyramid pyramid of a given image
    """
    gaussian_pyramid = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    pyr = []
    filter_vec = gaussian_1d(filter_size).reshape((1, filter_size))
    filter_vec = filter_vec / np.sum(filter_vec)

    for i in range(len(gaussian_pyramid) - 1):
        expanded_next_gaussian = expand(gaussian_pyramid[i+1], filter_vec)
        pyr.append(gaussian_pyramid[i] - expanded_next_gaussian)

    pyr.append(gaussian_pyramid[-1])

    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Reconstruction of an image from its Laplacian Pyramid.

    :param lpyr:        the laplacian pyramid.
    :param filter_vec:  the filter that used during construction of the pyramid.
    :param coeff:       is a vector of the same size is as the number of levels in the pyramid.
                        Before reconstructing the image each level i of the laplacian pyramid
                        is multiplied by its corresponding coefficient coeff[i].
    :return:            the reconstructed image.
    """
    temp_pyr = []

    for i in range(len(lpyr)):
        temp_pyr.append(coeff[i] * lpyr[i])

    for i in range(len(temp_pyr)-2, -1, -1):
        temp_pyr[i] = expand(temp_pyr[i+1], filter_vec) + temp_pyr[i]

    return temp_pyr[0]


def render_pyramid(pyr, levels):
    """
    Construct a single black image in which the pyramid levels of the
    given pyramid pyr are stacked horizontally (after stretching the values to [0, 1])

    :param pyr:     the given pyramid to render
    :param levels:  number of levels to render
    :return:        the constructed image
    """
    pyr_stretched = stretch_pyr(pyr)

    height, width = pyr_stretched[0].shape
    data_type = pyr_stretched[0].dtype

    im_pyr = np.empty((height, int((2-0.5**(levels-1)) * width)))

    for i in range(levels):
        curr_height, curr_width = int((0.5**i) * height), int((0.5**i) * width)
        temp = np.zeros((height, curr_width), dtype=data_type)
        temp[:curr_height, :curr_width] = pyr_stretched[i]
        temp[curr_height:, :] = 0

        curr_col_start = int((2 - 0.5**(i - 1)) * width)
        curr_col_end = int(curr_col_start + (0.5**i) * width)

        im_pyr[:, curr_col_start:curr_col_end] = temp

    return im_pyr


def display_pyramid(pyr, levels):
    """
    Display a single black image in which the pyramid levels of the
    given pyramid pyr are stacked horizontally (after stretching the values to [0, 1])

    :param pyr:     the given pyramid to display
    :param levels:  number of levels to display
    """
    plt.imshow(render_pyramid(pyr, levels), cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    pyramid blending as described in the lecture.

    :param im1:                  the first input grayscale image to be blended.
    :param im2:                  the second input grayscale image to be blended.
    :param mask:                 is a boolean mask containing True and False representing
                                  which parts of im1 and im2 should appear in the result
    :param max_levels:           the parameter used when generating the pyramids.
    :param filter_size_im:       the size of the Gaussian filter (an odd scalar that represents
                                  a squared filter) which  defining the filter used in the
                                  construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask:     the size of the Gaussian filter (an odd scalar that represents
                                  a squared filter) which defining the filter used in the
                                  construction of the Gaussian pyramid of mask.

    :return:                     the result
    """
    im1_laplacian_pyramid, filter_vec_im1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_laplacian_pyramid, filter_vec_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    mask_gaussian_pyramid, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float32),
                                                                    max_levels,
                                                                    filter_size_mask)

    blended_im_laplacian_pyramid = []
    for i in range(len(mask_gaussian_pyramid)):
        blended_im_laplacian_pyramid.append(
            mask_gaussian_pyramid[i] * im1_laplacian_pyramid[i] +
            (1 - mask_gaussian_pyramid[i]) * im2_laplacian_pyramid[i]
        )

    return np.clip(
        laplacian_to_image(
            blended_im_laplacian_pyramid,
            filter_vec_im1,
            np.ones((len(blended_im_laplacian_pyramid)))),
        0, 1)


def blend_and_show(im1, im2, mask):
    """
    gets 2 RGB images and a binary mask,
    blend each channel and show the result before returning it.
    """
    im_blended = np.dstack((
        pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 6, 7, 5),
        pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 6, 7, 5),
        pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 6, 7, 5),
    ))

    fig = plt.figure()
    fig.add_subplot(221)
    plt.imshow(im1)
    fig.add_subplot(222)
    plt.imshow(im2)
    fig.add_subplot(223)
    plt.imshow(mask, cmap=plt.cm.gray)
    fig.add_subplot(224)
    plt.imshow(im_blended)
    plt.show()

    return im_blended


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import color
from scipy.misc import imread as imread
from os import path


def linear_stretch(im):
    # minVal = im[np.nonzero(im)[0]]
    # maxVal = im[np.argmax(im)[0][0]]

    im_stretch = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
    # im_stretch = np.round(255 * (im - 0) / (255 - 0))
    return im_stretch.astype(np.float32)


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, plt.cm.gray)
    plt.show()


def render_pyramid(pyr, levels):
    num_row = pyr[0].shape[0]
    num_col = 0

    # make sure no out of range
    if levels > len(pyr):
        levels = len(pyr)

    for i in range(levels):
        # num_col = np.sum(np.arange(pyr[0].shape[1], pyr[-1].shape[1], 2 ** (-levels)))
        num_col += pyr[i].shape[1]

    res = np.zeros(shape=(num_row, num_col))

    current_col = 0
    for i in range(levels):
        res[0: pyr[i].shape[0], current_col: pyr[i].shape[1] + current_col] = linear_stretch(pyr[i])
        current_col += pyr[i].shape[1]

    return res


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filter_vec3 = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)

    lc = []
    for i in range(len(l1)):
        lc.append(Gm[i] * l1[i] + (1 - Gm[i]) * l2[i])

    return laplacian_to_image(lc, filter_vec1, np.ones((len(lc)))).clip(0, 1)


def laplacian_to_image(lpyr, filter_vec, coeff):
    cur_im = lpyr[-1] * coeff[-1]
    for i in range(2, len(lpyr) + 1):
        cur_im = expand(cur_im, filter_vec, lpyr[-i].shape) + (lpyr[-i] * coeff[-i])

    return cur_im



def build_laplacian_pyramid(im, max_levels, filter_size):
    guss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lapl_pyr = []
    ker = create_kernel(filter_size)
    for i in range(len(guss_pyr) - 1):
        expaned_im = expand(guss_pyr[i + 1], ker, guss_pyr[i].shape)
        lapl_pyr.append(guss_pyr[i] - expaned_im)

    lapl_pyr.append(guss_pyr[-1])
    return lapl_pyr, filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    actualLevels = 1
    smallestDim = 16
    pyr = [im]
    filter_vec = create_kernel(filter_size)

    # multiply by 2 because when getting inside we add an image that is smaller by half...
    while actualLevels < max_levels and (pyr[-1].shape[0] >= smallestDim * 2 and pyr[-1].shape[1] >= smallestDim * 2):
        pyr.append(reduce(pyr[-1], filter_size))
        actualLevels += 1

    return pyr, filter_vec


def reduce(im, filter_size):
    return subSumple(blurIm(im, create_kernel(filter_size)))


def expand(im, filter_vec, new_shape):
    kernel = filter_vec * 2.0
    return blurIm(zeroPadding(im, new_shape), kernel)


def blurIm(im, kernel):
    bluredIm = ndimage.filters.convolve(im, [kernel], mode='wrap')
    bluredIm = ndimage.filters.convolve(bluredIm, (kernel.T)[:, np.newaxis], mode='wrap')
    return bluredIm


def subSumple(im):
    return im[::2, ::2]


def zeroPadding(im, new_shape):
    padded_im = np.zeros(new_shape)
    padded_im[::2, ::2] = im
    return padded_im


def create_kernel(size):
    kernel = base_kernel = np.array([1, 1], dtype=np.int64)

    if size == 1:
        return np.array([1])

    for i in range(size - 2):
        kernel = np.convolve(base_kernel, kernel)

    total_kernel = np.sum(kernel)
    kernel = kernel.astype(np.float32) / total_kernel
    return kernel


def read_image(filename, representation):
    im = imread(filename)
    # tokenize
    if (np.amax(im) > 1):
        im_float = im.astype(np.float32)
        im_float /= 255
    else:
        im_float = im.astype(np.float32)

    if representation == 1:
        # Convert to gray
        if len(im.shape) < 3:
            # Already gray, no need to convert
            returnImage = im_float
        else:
            # Convert to gray scale
            im_gray = color.rgb2gray(im_float)
            im_gray = im_gray.astype(np.float32)
            returnImage = im_gray
    else:
        returnImage = im_float

    return returnImage


def blend_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    im_blend = np.zeros(shape=im1.shape)
    for i in range(im1.shape[2]):
        im_blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask,
                                             max_levels, filter_size_im, filter_size_mask)
    return im_blend


def plot_images(im1, im2, mask, im_blend):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.show()


def blending_example1():
    im1 = read_image(path.realpath("images//sunglasses_reflection.jpg"), 2)
    im2 = read_image(path.realpath("images//full_moon.jpg"), 2)
    mask = read_image(path.realpath("images//sunglasses_reflection_mask.png"), 1)
    # mask = (mask > 0).astype(np.bool)
    mask = mask.astype(np.bool)
    max_level = 6
    filter_size_im = 5
    filter_size_mask = 11

    im_blend = blend_rgb(im1, im2, mask, max_level, filter_size_im, filter_size_mask)
    plot_images(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend


def blending_example2():
    im1 = read_image(path.realpath("images//shark.jpg"), 2)
    im2 = read_image(path.realpath("images//fishes.jpg"), 2)
    mask = read_image(path.realpath("images//shark_mask.png"), 1)
    mask = (mask > 0).astype(np.bool)

    max_level = 5
    filter_size_im = 13
    filter_size_mask = 13

    im_blend = blend_rgb(im1, im2, mask, max_level, filter_size_im, filter_size_mask)
    plot_images(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend

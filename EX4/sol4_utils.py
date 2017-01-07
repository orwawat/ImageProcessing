import numpy as np
from scipy import ndimage
from scipy.misc import imread
from skimage import color
from scipy.signal import convolve2d

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
            im_gray = im_gray  # .astype(np.float32)
            returnImage = im_gray
    else:
        returnImage = im_float

    return returnImage  # .astype(np.float32)

def zeroPadding(im, new_shape):
    padded_im = np.zeros(new_shape)
    padded_im[::2, ::2] = im
    return padded_im

def reduce(im, filter_size):
    return subSumple(blur_spatial(im, filter_size))

def expand(im, filter_vec, new_shape):
    kernel = filter_vec * 2.0
    return blur_with_kernel(zeroPadding(im, new_shape), kernel)

def subSumple(im):
    return im[::2, ::2]

def create_kernel(size):
    kernel = base_kernel = np.array([1, 1], dtype=np.int64)

    if size == 1:
        return np.array([1])

    for i in range(size - 2):
        kernel = np.convolve(base_kernel, kernel)

    total_kernel = np.sum(kernel)
    kernel = kernel.astype(np.float32) / total_kernel
    return kernel.reshape(1, size)

def blur_with_kernel(im, kernel):
    if len(kernel.shape) == 1:
        kernel = kernel[:, np.newaxis]
    bluredIm = ndimage.filters.convolve(im, kernel, mode='reflect')
    bluredIm = ndimage.filters.convolve(bluredIm, kernel.T, mode='reflect')

    return bluredIm

def blur_spatial(im, kernel_size):
    kernel = create_kernel(kernel_size)
    return blur_with_kernel(im, kernel)

def laplacian_to_image(lpyr, filter_vec, coeff):
    cur_im = lpyr[-1] * coeff[-1]
    for i in range(2, len(lpyr) + 1):
        cur_im = expand(cur_im, filter_vec, lpyr[-i].shape) + (lpyr[-i] * coeff[-i])

    return cur_im.astype(np.float32)

def build_gaussian_pyramid(im, max_levels, filter_size):
    actualLevels = 1
    smallestDim = 16
    pyr = [im]
    filter_vec = create_kernel(filter_size)

    # multiply by 2 because when getting inside we add an image that is smaller by half...
    while actualLevels < max_levels and (pyr[-1].shape[0] >= smallestDim * 2 and pyr[-1].shape[1] >= smallestDim * 2):
        pyr.append(reduce(pyr[-1], filter_size).astype(np.float32))
        actualLevels += 1

    return pyr, filter_vec

def build_laplacian_pyramid(im, max_levels, filter_size):
    guss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lapl_pyr = []
    for i in range(0, len(guss_pyr) - 1):
        expaned_im = expand(guss_pyr[i + 1], filter_vec, guss_pyr[i].shape)
        lapl_pyr.append((guss_pyr[i] - expaned_im).astype(np.float32))

    lapl_pyr.append(guss_pyr[-1].astype(np.float32))

    return lapl_pyr, filter_vec

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filter_vec3 = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)

    lc = []
    for i in range(len(l1)):
        lc.append(Gm[i] * l1[i] + (1 - Gm[i]) * l2[i])

    return laplacian_to_image(lc, filter_vec1, np.ones((len(lc)))).clip(0, 1)

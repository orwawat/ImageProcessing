import numpy as np
from scipy import ndimage
from scipy.misc import imread as imread
from skimage import color



def laplacian_to_image(lpyr, filter_vec, coeff):
    cur_im = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 1):
        cur_im = expand(cur_im, filter_vec, lpyr[-i].shape) + (lpyr[-i] * coeff[-i])

    return cur_im

def build_laplacian_pyramid(im, max_levels, filter_size):
    guss_pyr, filter_vec = build_gaussian_pyramid(im,max_levels, filter_size)
    lapl_pyr = []
    for i in range(len(guss_pyr) - 1):
        expaned_im = expand(guss_pyr[i+1], create_kernel(filter_size), guss_pyr[i].shape)
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
    kernel = filter_vec * 2
    return blurIm(zeroPadding(im, new_shape), kernel)

def blurIm(im, kernel):
    bluredIm = ndimage.filters.convolve(im, [kernel], mode='wrap')
    bluredIm = ndimage.filters.convolve(bluredIm, (kernel.T)[:, np.newaxis], mode='wrap')
    return bluredIm

def subSumple(im):
    return im[::2, ::2]

def zeroPadding(im, new_shape):
    padded_im = np.zeros(new_shape)
    odd_col = np.arange(0, im.shape[0] * 2, 2)
    odd_row = np.arange(0, im.shape[1] * 2, 2)
    # TODO: WTF??
    padded_im[np.ix_(odd_col,odd_row)] = im
    # col_padded = np.insert(im, slice(a, None), 0, axis=1)
    # row_padded = np.insert(col_padded, slice(a, None), 0, axis=0)
    return padded_im


def create_kernel(size):
    kernel = base_kernel = np.array([1, 1], dtype=np.int64)

    for i in range(size - 2):
        kernel = np.convolve(base_kernel, kernel)
    # kernel = np.convolve(kernel, kernel.T)

    total_kernel = np.sum(kernel)
    kernel = kernel.astype(np.float32) / total_kernel
    return kernel
import sol3 as mySol
import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
import matplotlib.pyplot as plt

IM_NAME = 'lena.jpg'


def get_image():
    return get_specific_image(IM_NAME)


def get_specific_image(name):
    im = imread(".//images//" + name)

    im_gray = color.rgb2gray(im)
    im_gray = im_gray.astype(np.float32)
    # plt.imshow(im_gray, plt.cm.gray)
    # plt.show()
    return im_gray.astype(np.float32)


def show_plot(s_im):
    plt.imshow(s_im, plt.cm.gray)
    plt.show()


def test_build_gaussian_pyramid():
    im = get_image()
    max_levels = 3
    filter_size = 3

    pyr, filter_vec = mySol.build_gaussian_pyramid(im, max_levels, filter_size)
    if (len(filter_vec) != filter_size):
        print("XXXXXXXXXXXXXXX pyrmaid filter_ved len is wrong XXXXXXXXXXXXXXXXxx")
    if len(pyr) != max_levels:
        print("XXXXXXXXXXXXXXX pyrmaid length is wrong XXXXXXXXXXXXXXXXxx")
    # for pyr_im in pyr:
    #     show_plot(pyr_im)
    mySol.display_pyramid(pyr, max_levels)
    # make sure the smallest image isn't less than 16
    pyr, filter_vec = mySol.build_gaussian_pyramid(im, 1000, filter_size)
    if (pyr[-1].shape[0] < 16 or pyr[-1].shape[1] < 16):
        print("XXXXXXXXXXXXXXX pyrmaid smallest resloution is wrong XXXXXXXXXXXXXXXXxx")


def test_build_laplacian_pyramid():
    im = get_image()
    max_levels = 5
    filter_size = 9

    pyr, filter_vec = mySol.build_laplacian_pyramid(im, max_levels, filter_size)
    if (len(filter_vec) != filter_size):
        print("XXXXXXXXXXXXXXX pyramid filter_ved len is wrong XXXXXXXXXXXXXXXXxx")
    if len(pyr) != max_levels:
        print("XXXXXXXXXXXXXXX pyramid length is wrong XXXXXXXXXXXXXXXXxx")
    # for pyr_im in pyr:
    #     show_plot(pyr_im)
    mySol.display_pyramid(pyr, max_levels)
    # make sure the smallest image isn't less than 16
    pyr, filter_vec = mySol.build_gaussian_pyramid(im, 1000, filter_size)
    if (pyr[-1].shape[0] < 16 or pyr[-1].shape[1] < 16):
        print("XXXXXXXXXXXXXXX pyrmaid smallest resloution is wrong XXXXXXXXXXXXXXXXxx")


def test_laplacian_to_image():
    im = get_image()
    max_levels = 6
    filter_size = 9
    cofee = [1, 1, 1, 1, 1, 5]
    # cofee = [2,1,0.5,0.5,0.5,0.5]
    pyr, filter_vec = mySol.build_laplacian_pyramid(im, max_levels, filter_size)
    # mySol.display_pyramid(pyr, max_levels)
    actualIm = mySol.laplacian_to_image(pyr, filter_vec, cofee)

    mySol.display_pyramid([im, actualIm], 2)
    # show_plot(actualIm)
    diff = actualIm.flatten() - im.flatten()
    if not np.allclose(actualIm.flatten(), im.flatten(), atol=1.e-7):
        print("XXXXXXXXXXXXXXX pyramid recoustraction is wrong XXXXXXXXXXXXXXXXxx")
        ac = actualIm.flatten()
        iac = im.flatten()
        print(diff[diff.nonzero()])
        print('actual: \n', ac, "\n\n")
        print('excpected: \n', iac, "\n\n")

    else:
        print("VVVVVVVVVVVVVVVvv pyramid recoustraction is GOOOOOOOOD VVVVVVVVVVVVVVVVVVVV")


def test_sub_sample():
    array = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
    expectedArray = np.array([[1, 3], [8, 10]])
    actualArray = mySol.subSumple(array)
    if np.array_equal(expectedArray, actualArray):
        print("VVVVVVVVVVV passed test_sub_sample VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Failed test_sub_sample")
        print('expected: \n', expectedArray, "\n")
        print('actual: \n', actualArray, "\n")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


def test_zero_padding():
    array = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    expectedArray = np.array([[1, 0, 2, 0, 3, 0, 4], [0, 0, 0, 0, 0, 0, 0], [4, 0, 5, 0, 6, 0, 7]])
    actualArray = mySol.zeroPadding(array, expectedArray.shape)
    if np.array_equal(expectedArray, actualArray):
        print("VVVVVVVVVVV passed test_sub_sample VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Failed test_sub_sample")
        print('expected: \n', expectedArray, "\n")
        print('actual: \n', actualArray, "\n")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


def test_create_kernel():
    print("************* Start test_create_kernel: *************")
    ker_size = 5
    kernel = mySol.create_kernel(ker_size)
    print(kernel)


def create_2d_kernel(size):
    kernel = base_kernel = np.array([[1, 1]], dtype=np.int64)

    for i in range(size - 2):
        kernel = sig.convolve2d(base_kernel, kernel)
    kernel = sig.convolve2d(kernel, kernel.T)

    total_kernel = np.sum(kernel)
    kernel = kernel.astype(np.float32) / total_kernel
    return kernel


def test_blur_im():
    print("************* Start test_blur_im: *************")
    kernel_size = 3
    im = get_image()
    kernel = create_2d_kernel(kernel_size)
    expected_im = sig.convolve2d(im, kernel, mode='same', boundary='wrap').astype(np.float32)
    actual_im = mySol.blurIm(im, mySol.create_kernel(kernel_size))
    if np.allclose(expected_im, actual_im):
        print("VVVVVVVVVVV passed test_blur_im VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Failed test_blur_im")
        show_plot(expected_im)
        show_plot(actual_im)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


def test_blend():
    im1 = get_specific_image('car.jpg')
    im2 = get_specific_image('ball.jpg')
    mask = get_specific_image('car_mask.jpg')
    max_level = 3
    filter_size_im = 3
    filter_size_mask = 1

    blended = mySol.pyramid_blending(im1, im2, mask, max_level, filter_size_im, filter_size_mask)
    show_plot(blended)


def test_reduce():
    kernel_size = 3
    im = get_image()
    im_small = mySol.reduce(im, kernel_size)

    show_plot(im)
    show_plot(im_small)
    print(im.shape[0] == im_small.shape[0] * 2 and im.shape[1] == im_small.shape[1] * 2)


def test_readce_expand():
    im = get_image()
    kernel_size = 5
    im_small = mySol.reduce(im, kernel_size)
    im_expanded = mySol.expand(im_small, mySol.create_kernel(kernel_size), im.shape)

    mySol.display_pyramid([im, im_expanded, im_small], 3)


def test_display_pyramid():
    im = get_image()
    max_level = 6
    filter_size = 5
    pyr, filter_vec = mySol.build_gaussian_pyramid(im, max_level, filter_size)
    mySol.display_pyramid(pyr, len(pyr) - 1)

    pyr, filter_vec = mySol.build_laplacian_pyramid(im, max_level, filter_size)
    mySol.display_pyramid(pyr, len(pyr))


# test_readce_expand()
# mySol.blending_example1()
# mySol.blending_example2()
# test_blend()
# test_display_pyramid()
# test_build_gaussian_pyramid()
# test_build_laplacian_pyramid()
test_laplacian_to_image()
# test_create_kernel()
# test_sub_sample()
# test_zero_padding()
# test_blur_im()
# test_reduce()


# convert mask to bool
# mask_name = 'shark_mask.png'
# im = get_specific_image(mask_name)
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         if im[i, j] > 0:
#             im[i, j] = 1
#         else:
#             im[i, j] = 0
# # im_bool = im.astype(np.bool)
# imsave(".//images//" + 'bool_shark_mask.png', im)

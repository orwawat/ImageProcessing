import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
import matplotlib.pyplot as plt
import sol4 as mySol
from sol4_add import *
from sol4_utils import *

IM_NAME = 'backyard2.jpg'
# IM_NAME = 'backyard1.jpg'
MATCHING_IMAGES = ['backyard1.jpg', 'backyard2.jpg', 'backyard3.jpg']
# MATCHING_IMAGES = ['oxford1.jpg', 'oxford2.jpg']
# MATCHING_IMAGES = ['office1.jpg', 'office2.jpg', 'office3.jpg', 'office4.jpg']


def get_images():
    res = []
    for im in MATCHING_IMAGES:
        res.append(get_specific_image(im))
    return res


def get_image():
    return get_specific_image(IM_NAME)


def get_specific_image(name):
    im = imread(".//external//" + name)

    im_gray = color.rgb2gray(im)
    im_gray = im_gray.astype(np.float32)
    # plt.imshow(im_gray, plt.cm.gray)
    # plt.show()
    return im_gray.astype(np.float32)


def show_plot(s_im):
    plt.imshow(s_im, plt.cm.gray)
    plt.show()


def test_harris_corner_detector():
    ims = get_images()

    pos1 = spread_out_corners(ims[0], 7, 7, 3)
    im1 = build_gaussian_pyramid(ims[0], 3, 3)[0][2]
    im2 = build_gaussian_pyramid(ims[1], 3, 3)[0][2]
    pos2 = spread_out_corners(ims[1], 7, 7, 3)
    # pos1 = mySol.harris_corner_detector(ims[0])
    # pos2 = mySol.harris_corner_detector(ims[1])
    if pos1.shape[1] != 2:
        print("XXXXXXXXXXXXXXXXXX worng postion shape in: harris_corner_detector XXXXXXXXXXXXXXXXXXXX\n\n")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(ims[0], plt.cm.gray)
    plt.scatter(pos1[:, 0], pos1[:, 1])
    plt.subplot(2, 2, 2)
    plt.imshow(ims[1], plt.cm.gray)
    plt.scatter(pos2[:, 0], pos2[:, 1])
    plt.subplot(2, 2, 3)
    plt.imshow(im1, plt.cm.gray)
    plt.scatter(pos1[:, 0] / 4, pos1[:, 1] / 4)
    plt.subplot(2, 2, 4)
    plt.imshow(im2, plt.cm.gray)
    plt.scatter(pos2[:, 0] / 4, pos2[:, 1] / 4)
    plt.show()


def test_sample_descriptor():
    # TODO: test something :)
    im = get_image()
    radius = 3
    pos = mySol.harris_corner_detector(im)
    res = mySol.sample_descriptor(im, pos, radius)
    for i in range(res.shape[2]):
        # Print the descriptors of each point
        print('The descriptor of point: ', pos[i, :], ' is: ')
        print(res[:, :, i], '\n\n')


def test_descriptor_score2():
    ims = get_images()
    gaussian_blur = 3
    pyr1 = build_gaussian_pyramid(ims[0], 3, gaussian_blur)[0]
    pyr2 = build_gaussian_pyramid(ims[1], 3, gaussian_blur)[0]
    # pos1 = spread_out_corners(im1, 7, 7, 3)
    # desc1 = mySol.sample_descriptor(im1, pos1, 3)
    # desc2 = mySol.sample_descriptor(im1, spread_out_corners(im2, 7, 7, 3), 3)
    pos1, desc1 = mySol.find_features(pyr1)
    pos2, desc2 = mySol.find_features(pyr2)
    dotProduct = mySol.calc_descriptor_score(desc1, desc2)
    if (dotProduct.max() > 1 or dotProduct.min() < -1):
        print('XXXXXX descriptors dot product is not in [-1,1] range XXXXXXXXXX')
    else:
        print('OK')


def test_descriptor_score():
    ims = get_images()
    pos1 = spread_out_corners(ims[0], 7, 7, 3)
    desc1 = mySol.sample_descriptor(ims[0], pos1, 3)
    desc2 = mySol.sample_descriptor(ims[1], spread_out_corners(ims[1], 7, 7, 3), 3)
    dotProduct = mySol.calc_descriptor_score(desc1, desc2)
    if (dotProduct.max() > 1 or dotProduct.min() < -1):
        print('XXXXXX descriptors dot product is not in [-1,1] range XXXXXXXXXX')


def test_transform_coordinates_level():
    point = [10, 17]
    pos = np.array([point])
    current_level = 0
    wanted_level = 2
    actual = mySol.transform_coordinates_level(pos, current_level, wanted_level)

    expected_x = point[0]
    expected_y = point[1]
    for i in range(wanted_level - current_level):
        expected_x = expected_x / 2.0
        expected_y = expected_y / 2.0

    if actual[0, 0] != expected_x or actual[0, 1] != expected_y:
        print("XXXXXXX test_transform_coordinates_level XXXXXXXX")
    else:
        print("test_transform_coordinates_level - OK")


def test_match_features():
    ims = get_images()
    min_score = 0.5
    im1 = ims[0]
    im2 = ims[1]
    pos1, desc_1 = mySol.find_features(build_gaussian_pyramid(im1, 3, 3)[0])
    pos2, desc_2 = mySol.find_features(build_gaussian_pyramid(im2, 3, 3)[0])
    match_ind1, match_ind2 = mySol.match_features(desc_1, desc_2, min_score)

    for i in range(len(match_ind1)):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im1, plt.cm.gray)
        plt.scatter(pos1[match_ind1[i], 0], pos1[match_ind1[i], 1])
        plt.subplot(1, 2, 2)
        plt.imshow(im2, plt.cm.gray)
        plt.scatter(pos2[match_ind2[i], 0], pos2[match_ind2[i], 1])
        plt.show()




# test_harris_corner_detector()
# test_sample_descriptor()
# test_transform_coordinates_level()
# test_descriptor_score()
# test_descriptor_score2()
test_match_features()

# twod = np.array([[0,0], [1,1], [2,2], [3,3]])
# a = np.array([twod, twod])
# print(mySol.match_features(a,a,0))
# b = a.reshape((a.shape[2], -1), order='F')
# print(b)
# print(a)

x = [1, 2, 3]
y = [2, 3, 4]


# print(np.dstack(np.meshgrid(x, y)).reshape(-1, 2))

import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
from timeit import timeit
import matplotlib.pyplot as plt
import sol4 as mySol
from sol4_add import *
from sol4_utils import *

IM_NAME = 'backyard2.jpg'
# IM_NAME = 'backyard1.jpg'
MATCHING_IMAGES = ['backyard1.jpg', 'backyard2.jpg']#, 'backyard3.jpg']


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
    im = color.rgb2gray(im)
    im_gray = im.astype(np.float32)
    # plt.imshow(im_gray, plt.cm.gray)# plt.show()
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
    # plt.subplot(2, 2, 3)
    # plt.imshow(im1, plt.cm.gray)
    # plt.scatter(pos1[:, 0] / 4, pos1[:, 1] / 4)
    # plt.subplot(2, 2, 4)
    # plt.imshow(im2, plt.cm.gray)
    # plt.scatter(pos2[:, 0] / 4, pos2[:, 1] / 4)
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
    im1 = ims[1]
    im2 = ims[2]
    gaussian_blur = 3
    pyr1 = build_gaussian_pyramid(im1, 3, gaussian_blur)[0]
    pyr2 = build_gaussian_pyramid(im2, 3, gaussian_blur)[0]
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
    im1 = ims[1]
    im2 = ims[2]
    m = 5
    n = 5
    radius = 5
    pos1 = spread_out_corners(im1, m, n, radius)
    desc1 = mySol.sample_descriptor(im1, pos1, 3)
    desc2 = mySol.sample_descriptor(im2, spread_out_corners(im2, m, n, radius), 3)
    dotProduct = mySol.calc_descriptor_score(desc1, desc2)
    if (dotProduct.max() > 1 or dotProduct.min() < -1):
        print('XXXXXX descriptors dot product is not in [-1,1] range XXXXXXXXXX')
    else:
        print('OK')


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


def test_ransac_homography():
    ims = get_images()
    min_score = 0.5
    im1 = ims[0]
    im2 = ims[1]

    pos1, desc_1 = mySol.find_features(build_gaussian_pyramid(im1, 3, 3)[0])
    pos2, desc_2 = mySol.find_features(build_gaussian_pyramid(im2, 3, 3)[0])
    match_ind1, match_ind2 = mySol.match_features(desc_1, desc_2, min_score)
    H12, inliners = mySol.ransac_homography(np.take(pos1, match_ind1, 0), np.take(pos2, match_ind2, 0), 10000, 6)
    mySol.display_matches(im1, im2, np.take(pos1, match_ind1, 0), np.take(pos2, match_ind2, 0), inliners)
    # mySol.display_matches(im1, im2, pos1, pos2, inliners)
    if (type(inliners) != np.ndarray):
        print('XXXXXX Wrong type')
    elif (H12.shape != (3, 3)):
        print('XXXXXX Wrong shape')
    else:
        print('OK')


def test_get_highest_indices():
    arr1 = np.array([0.5,None,0.99,0.99])
    print(arr1[mySol.get_highest_indices(arr1)])


def test_match_features1():
    arr1 = []
    arr2 = []
    mySol.match_features(arr1, arr2, 0.5)


def test_match_features():
    ims = get_images()
    min_score = 0.9
    im1 = ims[0]
    im2 = ims[1]
    pos1, desc_1 = mySol.find_features(build_gaussian_pyramid(im1, 3, 3)[0])
    pos2, desc_2 = mySol.find_features(build_gaussian_pyramid(im2, 3, 3)[0])
    match_ind1, match_ind2 = mySol.match_features(desc_1, desc_2, min_score)

    for i in range(35, len(match_ind1)):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im1, plt.cm.gray)
        plt.scatter(pos1[match_ind1[i], 0], pos1[match_ind1[i], 1])
        plt.subplot(1, 2, 2)
        plt.imshow(im2, plt.cm.gray)
        plt.scatter(pos2[match_ind2[i], 0], pos2[match_ind2[i], 1])
        plt.show()


def test_apply_homography():
    H12 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    pos1 = np.array([[1, 2], [2, 3], [5, 0], [7, 7]])
    pos2 = mySol.apply_homography(pos1, H12)

    if (pos2.shape != pos1.shape):
        print('XXXXXX test_apply_homography bad shape XXXXXXX')
    elif not np.array_equal(pos1, pos2):
        print('XXXXXX test_apply_homography bad values XXXXXXX')
    else:
        print('OK')


def test_accumulate_homographies_3():
    # num_samples = 5
    # H_successive = [None] * num_samples
    # for i in range(num_samples):
    #     H_successive[i] = np.random.random_sample(9).reshape(3, 3)

    H_successive = []
    # H_successive.append(np.ones(shape=(3,3)) * 2)
    # H_successive.append(np.ones(shape=(3, 3)))
    # H_successive.append(np.ones(shape=(3, 3)) * 0.5)
    # H_successive.append(np.ones(shape=(3, 3)) * 3)

    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))

    m = 3
    excepected_h2m = [None] * (len(H_successive) + 1)
    excepected_h2m[0] = H_successive[0] * H_successive[1] * H_successive[2]
    excepected_h2m[1] = H_successive[1] * H_successive[2]
    excepected_h2m[2] = H_successive[2]
    excepected_h2m[3] = np.eye(3)
    excepected_h2m[4] = np.linalg.inv(H_successive[3])

    for mat in excepected_h2m:
        mat /= mat[2, 2]

    actual_h2m = mySol.accumulate_homographies(H_successive, m)
    if (np.allclose(actual_h2m, excepected_h2m)):
        print('OK')
    else:
        print('XXXXXXX test_accumulate_homographies XXXXXXX')


def test_accumulate_homographies_0():
    H_successive = []
    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))
    H_successive.append(np.random.random_sample(9).reshape(3, 3))

    m = 0
    excepected_h2m = [None] * (len(H_successive) + 1)
    excepected_h2m[0] = np.eye(3)
    # H_successive[0] * H_successive[1] * H_successive[2]
    excepected_h2m[1] = np.linalg.inv(H_successive[0])
    excepected_h2m[2] = np.linalg.inv(H_successive[0]) * np.linalg.inv(H_successive[1])
    excepected_h2m[3] = np.linalg.inv(H_successive[0]) * np.linalg.inv(H_successive[1]) * np.linalg.inv(H_successive[2])
    excepected_h2m[4] = np.linalg.inv(H_successive[0]) * np.linalg.inv(H_successive[1]) * np.linalg.inv(
        H_successive[2]) * np.linalg.inv(H_successive[3])

    for mat in excepected_h2m:
        mat /= mat[2, 2]

    actual_h2m = mySol.accumulate_homographies(H_successive, m)
    if (np.allclose(actual_h2m, excepected_h2m)):
        print('OK')
    else:
        print('XXXXXXX test_accumulate_homographies_0 XXXXXXX')


def test_get_empty_panorama():
    ims = get_images()
    min_score = 0.9
    # H_successive = [None] * (len(ims) - 1)
    H_successive = []

    for i in range(len(ims) - 1):
        pos1, desc_1 = mySol.find_features(build_gaussian_pyramid(ims[i], 3, 3)[0])
        pos2, desc_2 = mySol.find_features(build_gaussian_pyramid(ims[i + 1], 3, 3)[0])
        match_ind1, match_ind2 = mySol.match_features(desc_1, desc_2, min_score)
        H_successive.append(
            mySol.ransac_homography(np.take(pos1, match_ind1, 0), np.take(pos2, match_ind2, 0), 1000, 3)[0])

    Hs = mySol.accumulate_homographies(H_successive, len(ims) // 2)

    points = mySol.get_empty_panroma(ims, Hs)
    print(points)


def test_render_panorama():
    ims = get_images()
    min_score = 0.9
    # H_successive = [None] * (len(ims) - 1)
    H_successive = []

    for i in range(len(ims) - 1):
        pos1, desc_1 = mySol.find_features(build_gaussian_pyramid(ims[i], 3, 3)[0])
        pos2, desc_2 = mySol.find_features(build_gaussian_pyramid(ims[i + 1], 3, 3)[0])
        match_ind1, match_ind2 = mySol.match_features(desc_1, desc_2, min_score)
        H_successive.append(
            mySol.ransac_homography(np.take(pos1, match_ind1, 0), np.take(pos2, match_ind2, 0), 10000, 6)[0])

    Hs = mySol.accumulate_homographies(H_successive, (len(ims) - 1) // 2)
    panorma = mySol.render_panorama(ims, Hs)
    plt.imshow(panorma, plt.cm.gray)
    # plt.imshow(panorma)
    plt.show()


# test_harris_corner_detector()
# test_sample_descriptor()
# test_transform_coordinates_level()
# test_descriptor_score()
# test_descriptor_score2()
test_match_features()
# test_apply_homography()
# test_ransac_homography()
# test_accumulate_homographies_3()
# test_accumulate_homographies_0()
# test_get_highest_indices()

import time

t0 = time.time()
# test_render_panorama()
t1 = time.time()

print(t1 - t0)

# twod = np.array([[0,0], [1,1], [2,2], [3,3]])
# a = np.array([twod, twod])
# print(mySol.match_features(a,a,0))
# b = a.reshape((a.shape[2], -1), order='F')
# print(b)
# print(a)

# print(timeit("test_render_panorama()", setup='from __main__ import test_render_panorama',number = 100))
# print(timeit("np.empty(shape=(1000,10))", setup='import numpy as np', number=1000))
# print(timeit("[None] * 1000", number=1000))

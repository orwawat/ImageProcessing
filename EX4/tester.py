import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
import matplotlib.pyplot as plt
import sol4 as mySol
from sol4_add import *

# IM_NAME = 'office2.jpg'
IM_NAME = 'backyard1.jpg'


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
    im = get_image()
    pos = spread_out_corners(im, 7, 7, 7)
    # pos = mySol.harris_corner_detector(im)
    if pos.shape[1] != 2:
        print("XXXXXXXXXXXXXXXXXX worng postion shape in: harris_corner_detector XXXXXXXXXXXXXXXXXXXX\n\n")

    plt.imshow(im, plt.cm.gray)
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


def test_sample_descriptor():
    # TODO: test something :)
    im = get_image()
    radius = 3
    res = mySol.sample_descriptor(im, mySol.harris_corner_detector(im), radius)
    print(res)

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

    if actual[0,0] != expected_x or actual[0,1] != expected_y:
        print("XXXXXXX test_transform_coordinates_level XXXXXXXX")
    else:
        print("test_transform_coordinates_level - OK")

# test_harris_corner_detector()
test_sample_descriptor()
# test_transform_coordinates_level()

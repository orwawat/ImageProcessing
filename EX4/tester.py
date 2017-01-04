import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
import matplotlib.pyplot as plt
import sol4 as mySol

IM_NAME = 'office2.jpg'


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
    pos = mySol.harris_corner_detector(im)

    if pos.shape[1] != 2:
        print("XXXXXXXXXXXXXXXXXX worng postion shape in: harris_corner_detector XXXXXXXXXXXXXXXXXXXX\n\n")

    plt.imshow(im, plt.cm.gray)
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


test_harris_corner_detector()

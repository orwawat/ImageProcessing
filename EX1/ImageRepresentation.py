import numpy as np
from scipy.misc import imread as imread, imsave as imsave
from skimage import data, io, filters, color


class RgbYiqConverter:
    yiqToRgb = np.array([[0.299, 0.587, 0.114],
                            [0.596, -0.275, -0.321],
                            [0.212, -0.523, 0.311]])

    @staticmethod
    def getYIQ(rgbImg):
        return RgbYiqConverter.yiqToRgb.dot(rgbImg)

    @staticmethod
    def getRGB(yiqImg):
        return np.linalg.inv(RgbYiqConverter.yiqToRgb).dot(yiqImg)

def read_image(filename, representation):
    im = imread(filename)
    im_float = im.astype(np.float32)
    im_float /= 255

    if im.shape >= 3 and representation == 1:
        im_gray = color.rgb2gray(im_float)
        im_gray = im_gray.astype(np.float32)
        returnImage = im_gray
    else:
        returnImage = im_float

    return returnImage


def imdisplay(filename, representation):
    io.imshow(read_image(filename, representation))
    io.show()


def rgb2yiq(imRGB):
    return RgbYiqConverter.getYIQ(imRGB)

def yiq2rgb(imYIQ):
    return RgbYiqConverter.getRGB(imYIQ)

print(rgb2yiq(read_image('color.jpg', 2)))
imdisplay('color.jpg', 2)

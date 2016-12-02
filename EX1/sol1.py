import numpy as np
from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt
from skimage import data, io, filters, color


MAX_COLOR = 255
class RgbYiqConverter:
    yiqToRgb = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])

    @staticmethod
    def getYIQ(rgbImg):
        # Convert to 3 row vector's
        redPic = rgbImg[:, :, 0].ravel()
        greenPic = rgbImg[:, :, 1].ravel()
        bluePic = rgbImg[:, :, 2].ravel()
        # Convert to 3*1 matrix
        imgToConvert = np.row_stack((redPic, greenPic, bluePic))
        yiq = np.dot(RgbYiqConverter.yiqToRgb, imgToConvert)
        yiq = yiq.T
        yiq = yiq.reshape((rgbImg.shape[0], rgbImg.shape[1], rgbImg.shape[2]))
        return yiq.astype(np.float32)

    @staticmethod
    def getRGB(yiqImg):
        # Convert to 3 row vector's
        yPic = yiqImg[:, :, 0].ravel()
        iPic = yiqImg[:, :, 1].ravel()
        qPic = yiqImg[:, :, 2].ravel()
        # Convert to 3*1 matrix
        imgToConvert = np.row_stack((yPic, iPic, qPic))
        rgb = np.dot(np.linalg.inv(RgbYiqConverter.yiqToRgb), imgToConvert)
        rgb = rgb.T
        rgb = rgb.reshape((yiqImg.shape[0], yiqImg.shape[1], yiqImg.shape[2]))
        return rgb.astype(np.float32)


def isGraySacle(im):
    return len(im.shape) < 3

def getGrayImage(im_float):
    '''
    :param im_float: image astype=np.float32, values are in [0,1]
    :return:
    '''
    if isGraySacle(im_float):
        # Already gray, no need to convert
        grayImage = im_float
    else:
        # Convert to gray scale
        im_gray = color.rgb2gray(im_float)
        im_gray = im_gray.astype(np.float32)
        grayImage = im_gray

    return grayImage

def read_image(filename, representation):
    im = imread(filename)
    # tokenize
    im_float = im.astype(np.float32)
    im_float /= MAX_COLOR

    if representation == 1:
        # Convert to gray
        returnImage = getGrayImage(im_float)
    else:
        returnImage = im_float

    return returnImage

def imdisplay(filename, representation):
    arrdisplay(read_image(filename, representation))


def arrdisplay(im, representation):
    if representation == 1:
        plt.imshow(im, plt.cm.gray)
    else:
        plt.imshow(im)
    plt.show()

def rgb2yiq(imRGB):
    return RgbYiqConverter.getYIQ(imRGB)

def yiq2rgb(imYIQ):
    return RgbYiqConverter.getRGB(imYIQ)


def histogram_equalize(im_orig):
    if isGraySacle(im_orig):
        bw_im = im_orig
    else:
        yiq_im = rgb2yiq(im_orig)
        yiq_im = np.clip(yiq_im.astype(np.float32), 0, 1)
        bw_im = yiq_im[:, :, 0]

    hist_orig, bin_edges = np.histogram(bw_im, bins=MAX_COLOR + 1, range=(0, 1))
    hist_orig[0] = 50
    sumHist = np.cumsum(hist_orig)

    sumHist = (sumHist / (50 + bw_im.shape[0] * bw_im.shape[1]) * MAX_COLOR).astype(np.float32)
    minGray = sumHist.item(np.nonzero(sumHist)[0][0])

    # Starch the look up table
    lut = np.round(MAX_COLOR * (sumHist - minGray) / (sumHist.item(MAX_COLOR) - minGray))
    lut = lut.astype(np.int)


    bw_im_eq = lut[(bw_im * MAX_COLOR).astype(np.int)]
    hist_eq, bin_edges = np.histogram(bw_im_eq, bins=MAX_COLOR + 1)

    bw_im_eq = bw_im_eq.astype(np.float32) / MAX_COLOR
    # Convert back to rgb if needed
    if not isGraySacle(im_orig):
        bw_im_eq = convertYChannelToRgb(yiq_im, bw_im_eq)

    return [bw_im_eq, hist_orig, hist_eq]


def convertYChannelToRgb(im_orig, y_channel):
    im = np.stack((y_channel, im_orig[:, :, 1], im_orig[:, :, 2]), axis=-1)
    im = yiq2rgb(im)
    im = np.clip(im.astype(np.float32), 0, 1)
    return im


def findBorders(intensities, n_quant):
    borders = np.ndarray(shape=n_quant + 1, dtype=np.int)
    borders.itemset(0, 0)
    borders.itemset(n_quant, MAX_COLOR + 1)

    for i in range(0, n_quant - 1):
        borders.itemset(i + 1, np.around((intensities.item(i) + intensities.item(i + 1)) / 2))
    return borders


def findIntesities(borders, histogram, n_quant):
    intensities = np.ndarray(shape=n_quant)

    for i in range(0, n_quant):
        cur_hist_vals = histogram[borders[i]: borders[i + 1]]
        denominator = np.sum(cur_hist_vals)
        axis_values = np.linspace(borders[i], borders[i + 1], num=borders[i+1]-borders[i], endpoint=False, dtype=np.int)
        numerator = np.sum(axis_values * cur_hist_vals)

        intensities.itemset(i, np.around(numerator / denominator))
    return intensities

def calcSSD(intensities, borders, histogram):

    error = 0
    # sum on k-1 intensities
    for i in range(0, len(intensities)):
        # get all the values from z_i to z_i_1
        axis_values = np.linspace(borders[i], borders[i + 1], num=borders[i + 1] - borders[i], endpoint=False,
                                  dtype=np.int)
        cur_error = np.sum(np.square(intensities[i] - axis_values) * histogram[axis_values])
        error += cur_error

    return error

def intialBorders(shape, hist, n_quant):
    borders = np.zeros(n_quant + 1, dtype=np.int)
    total_pixels = shape[0] * shape[1]
    section_pixels = np.around(total_pixels / n_quant)
    borders[n_quant] = MAX_COLOR + 1

    sumHist = np.cumsum(hist)

    for i in range(1, n_quant):
        index = np.argmax(sumHist >= section_pixels * i)
        borders.itemset(i, index)

    return borders

def quantize(im_orig, n_quant, n_iter):
    error = []
    if isGraySacle(im_orig):
        bw_im = im_orig
    else:
        yiq_im = rgb2yiq(im_orig)
        yiq_im = np.clip(yiq_im.astype(np.float32), 0, 1)
        bw_im = yiq_im[:, :, 0]

    bw_im = np.around(bw_im * MAX_COLOR).astype(np.int)
    hist, bins = np.histogram(bw_im, bins=np.arange(MAX_COLOR + 2))

    index = 0
    convergence = False
    while index < n_iter and not convergence:
        if index == 0:
            segBorders = intialBorders(bw_im.shape, hist, n_quant)
            segIntensities = findIntesities(segBorders, hist, n_quant)
            prevBorder = 0
        else:
            prevBorder = segBorders
            segBorders = findBorders(segIntensities, n_quant)
            segIntensities = findIntesities(segBorders, hist, n_quant)

        convergence = np.array_equal(prevBorder, segBorders)
        if not convergence:
            error.append(calcSSD(segIntensities, segBorders, hist))

        index += 1

    # build a look up table
    lut = np.zeros(MAX_COLOR + 1)
    for i in range(0, len(segBorders) - 1):
        lut[segBorders[i]: segBorders[i + 1]] = segIntensities[i]
    lut[-1] = segIntensities[-1]

    im_quant = lut[bw_im]

    # convert back to color image
    if not isGraySacle(im_orig):
        im_quant = im_quant.astype(np.float32) / MAX_COLOR
        im_quant = convertYChannelToRgb(yiq_im, im_quant)

    return [im_quant, error]


histogram_equalize(read_image('./color.jpg', 1))
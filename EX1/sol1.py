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
        # TODO: return array in [0,255]?
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

def getYiqImage(im):
    if not isGraySacle(im):
        im = rgb2yiq(im)[:, :, 0]

    return im

def histogram_equalize(im_orig):
    im = getYiqImage(im_orig)

    im = (im * MAX_COLOR).astype(np.int)
    hist_orig, bin_edges = np.histogram(im, bins=256, range=(0, 256))
    sumHist = np.cumsum(hist_orig, dtype=np.float32)

    lut = sumHist / (im.shape[0] * im.shape[1]) * MAX_COLOR

    minGray = sumHist.item(np.nonzero(sumHist)[0][0])

    # Starch the look up table
    lut = np.round(MAX_COLOR * (sumHist - minGray) / (sumHist.item(MAX_COLOR) - minGray))

    lut = lut.astype(np.int)

    im_eq = lut[im]

    hist_eq, bin_edges = np.histogram(im_eq, bins=256, range=(0, 256))
    return [im_eq, hist_orig, hist_eq]


def findBorders(intensities, n_quant):
    borders = np.ndarray(shape=n_quant + 1)
    borders.itemset(0,0)
    borders.itemset(n_quant + 1, MAX_COLOR)

    for i in range(0,n_quant - 1):
        borders.itemset(i + 1, (intensities.item(i) + intensities.item(i + 1)) / 2)

def findIntesities(borders, histogram, n_quant):
    intensities = np.ndarray(shape=n_quant)
    for i in range(0,n_quant):
        numerator, denominator = 0
        for k in range(borders.item(i), borders.item(i + 1)):
            numerator += k * histogram.item(k)
            denominator += histogram.item(k)
        intensities.itemset(i, numerator / denominator)

def calcSSD(intensities, borders, histogram):
    error = 0
    for intens in intensities:
        for xIndex in range(borders[intens], borders[intens + 1]):
            error += np.power((intens - xIndex), 2) * histogram(xIndex)

    return error


def quantize (im_orig, n_quant, n_iter):
    error = np.zeros(n_quant)
    im = getYiqImage(im_orig)
    histogram = np.histogram(im, bins=MAX_COLOR + 1, range=(0, MAX_COLOR + 1))

    segBorders = np.linspace(0, MAX_COLOR, num=(n_quant + 1), endpoint=True, retstep=False, dtype=np.int)
    segIntensities = findIntesities(segBorders)
    i = 0
    convergence = False
    while i < n_iter or convergence:
        error.itemset(i, calcSSD(segIntensities, segBorders, histogram))
        segBorders = findBorders(segIntensities, n_quant)
        segIntensities = findIntesities(segBorders, histogram, n_quant)
        if i > 0:
            convergence = error.item(i) >= error.item(i - 1)

    for i in range(0,n_quant):
        im_quant = segIntensities[im[segBorders.item(i) : segBorders.item(i+1), :]]

    return [im_quant, error]

myPic = read_image('bw.jpg', 2)
#myPic = myPic[0:2,0:2,:].round(3)
myPic, hist_orig, hist_eq = histogram_equalize(myPic)
# arrdisplay(myPic, 1)

# plt.bar(hist_orig, 'b', hist_eq, 'r')
# plt.show()
#print("myPic is: \n")
#print(myPic)
#print("\n\n")

# yiqPic = rgb2yiq(myPic)
# print("first pic some pixel value: ", myPic[95, 60, :], "\n\n")
# print("yiq pic some pixel value: ", yiqPic[95, 60, :], "\n\n")
# rgb = yiq2rgb(yiqPic)
# print("rgb pic some pixel value: ", rgb[95, 60, :], "\n\n")
#print("********************************")
#print("myPic YIQ pic: \n")
#print(yiqPic)


#plt.imshow(yiqPic[:, :, 0], plt.cm.gray)

#imdisplay('color.jpg', 1)


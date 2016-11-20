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
        yiqIm = rgb2yiq(im_orig)
        bw_im = yiqIm[:, :, 0]

    bw_im = (bw_im * MAX_COLOR).astype(np.int)
    hist_orig, bin_edges = np.histogram(bw_im, bins=MAX_COLOR + 1, range=(0, MAX_COLOR + 1))
    sumHist = np.cumsum(hist_orig, dtype=np.float32)

    sumHist = (sumHist / (bw_im.shape[0] * bw_im.shape[1]) * MAX_COLOR).astype(np.float32)
    minGray = sumHist.item(np.nonzero(sumHist)[0][0])

    # Starch the look up table
    lut = np.round(MAX_COLOR * (sumHist - minGray) / (sumHist.item(MAX_COLOR) - minGray))
    lut = lut.astype(np.int)
    print("lut is: /n/n", lut)


    bw_im_eq = lut[bw_im]
    hist_eq, bin_edges = np.histogram(bw_im_eq, bins=MAX_COLOR + 1, range=(0, MAX_COLOR + 1))

    bw_im_eq = bw_im_eq.astype(np.float32) / MAX_COLOR
    # Convert back to rgb if needed
    if not isGraySacle(im_orig):
        bw_im_eq = convertYChannelToRgb(yiqIm, bw_im_eq)

    return [bw_im_eq, hist_orig, hist_eq]


def convertYChannelToRgb(im_orig, y_channel):
    im = np.stack((y_channel, im_orig[:, :, 1], im_orig[:, :, 2]), axis=-1)
    im = yiq2rgb(im)
    im = np.clip(im.astype(np.float32), 0, 1)
    return im


def findBorders(intensities, n_quant):
    borders = np.ndarray(shape=n_quant + 1, dtype=np.int)
    borders.itemset(0, 0)
    borders.itemset(n_quant, MAX_COLOR)

    for i in range(0, n_quant - 1):
        borders.itemset(i + 1, np.round((intensities.item(i) + intensities.item(i + 1)) / 2))
    return borders


def findIntesities(borders, histogram, n_quant):
    intensities = np.ndarray(shape=n_quant, dtype=np.uint8)
    for i in range(0, n_quant):
        cur_hist_vals = histogram[borders[i]: borders[i + 1]]
        denominator = np.sum(cur_hist_vals)
        axis_values = np.linspace(borders[i], borders[i + 1], num=borders[i+1]-borders[i], endpoint=False, dtype=np.int)
        numerator = np.sum(axis_values * cur_hist_vals)

        intensities.itemset(i, numerator // denominator)
    return intensities

def calcSSD(intensities, borders, histogram):
    error = 0

    for i in range(0, len(intensities)):
        axis_values = np.linspace(borders[i], borders[i + 1], num=borders[i + 1] - borders[i], endpoint=False,
                                  dtype=np.int)
        p = np.power((intensities.item(i) - axis_values), 2)
        error += np.sum(p * histogram[axis_values])

    return error

def intialBorders(im, hist, n_quant):
    borders = np.zeros(n_quant + 1, dtype=np.int)
    total_pixels = im.shape[0] * im.shape[1]
    section_pixels = total_pixels / n_quant
    borders[n_quant] = MAX_COLOR

    sumHist = np.cumsum(hist)

    for i in range(1, n_quant):
        index = np.argmax(sumHist >= section_pixels * i)
        borders.itemset(i, index)

    return borders

def quantize(im_orig, n_quant, n_iter):
    error = np.zeros(n_iter)
    if isGraySacle(im_orig):
        bw_im = im_orig
    else:
        bw_im = rgb2yiq(im_orig)
        bw_im = bw_im[:, :, 0]

    bw_im = (np.round(bw_im * 255)).astype(np.int)
    hist, bin_edges = np.histogram(bw_im, bins=np.arange(257))

    segBorders = intialBorders(bw_im, hist, n_quant)
    segIntensities = findIntesities(segBorders, hist, n_quant)

    i = 1
    convergence = False
    while i < n_iter and not convergence:

        error.itemset(i, calcSSD(segIntensities, segBorders, hist))
        print("**************************************************")
        print("The borders are: ", segBorders)
        print("**************************************************")
        print("**************************************************")
        print("The Intensities are: ", segIntensities)
        print("**************************************************")
        print("**************************************************")
        print("The error is: ", error.item(i))
        print("**************************************************")

        segIntensities = findIntesities(segBorders, hist, n_quant)
        prevBorder = segBorders
        segBorders = findBorders(segIntensities, n_quant)

        convergence = np.array_equal(prevBorder, segBorders)
        i += 1

    # build a look up table
    lut = np.zeros(MAX_COLOR + 1)
    for i in range(0, len(segBorders) - 1):
        lut[segBorders[i]: segBorders[i + 1]] = segIntensities[i]
    lut[-1] = segIntensities[-1]

    lut = lut.astype(np.int)
    im_quant = lut[bw_im]

    # convert back to color image
    if not isGraySacle(im_orig):
        im_quant = im_quant.astype(np.float32) / MAX_COLOR
        im_quant = convertYChannelToRgb(im_orig, im_quant)

    return [im_quant, error]



# myPic = read_image('.\\test\external\jerusalem.jpg', 2)

myPic = read_image('.//tester_files//monkey.jpg', 1)

'''
print("##################################################")
print("The Pic is: ", myPic)
print("##################################################")
'''

# myPic = read_image('color.jpg', 2)
myPic, error = quantize(myPic, 3, 4)
arrdisplay(myPic, 1)
plt.plot(error)
plt.show()


'''
myPic = read_image('.//tester_files//compare_files//rgb//equalized//Low Contrast.jpg', 2)

myPic, hist_orig, hist_eq = histogram_equalize(myPic)
arrdisplay(myPic, 2)
plt.plot(hist_orig, 'b', hist_eq, 'r')
plt.show()
'''

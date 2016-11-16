import numpy as np
from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt
from skimage import data, io, filters, color


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
    im_float /= 255

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


def histogram_equalize1(im_orig):

    if not isGraySacle(im_orig):
        im = yiq2rgb(im_orig)[:, :, 0]
    else:
        im = im_orig

    im = im * 256
    hist_orig, bin_edges = np.histogram(im, bins=256, range=(0, 256))
    # print("histogram is: ", hist_orig)
    # print("\n\n****************************\n\n")
    sumHist = np.cumsum(hist_orig, dtype=np.float32)
    # print("sumHist is: ", sumHist)
    # print("\n\n****************************\n\n")

    minGray = sumHist.item(np.nonzero(sumHist)[-1][-1])
    print(minGray)

    cdf_m = np.ma.masked_equal(sumHist, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (255 - cdf_m.min())
    lut = np.ma.filled(cdf_m, 0).astype('uint8')

    '''
    lut = np.ndarray(shape=(256))

    # print(sumHist[minGray])

    lut = np.round(255 * (sumHist - minGray) / sumHist.item(255) - minGray)

    for i in range(0, 256):
        print(np.round(255 * (sumHist[i] - sumHist[minGray]) / sumHist[255] - sumHist[minGray]))
        lut.itemset(i, np.round(255 * (sumHist[i] - sumHist[minGray]) / sumHist[255] - sumHist[minGray]))
        '''

    print(lut)
    im_eq = lut[im]

    hist_eq = np.histogram(im_eq, bins=256, range=(0,256))
    if not isGraySacle(im_orig):
        im_eq = yiq2rgb(im_eq)

    #print("The sum histogram is: ", sumHist)

    return [im_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    if not isGraySacle(im_orig):
        im = rgb2yiq(im_orig)[:, :, 0]
    else:
        im = im_orig

    im = (im * 255).astype(np.int)
    hist_orig, bin_edges = np.histogram(im, bins=256, range=(0, 256))
    sumHist = np.cumsum(hist_orig, dtype=np.float32)

    lut = sumHist / (im.shape[0] * im.shape[1]) * 255

    minGray = sumHist.item(np.nonzero(sumHist)[0][0])

    # Starch the look up table
    lut = np.round(255 * (sumHist - minGray) / (sumHist.item(255) - minGray))

    lut = lut.astype(np.int)

    im_eq = lut[im]

    hist_eq, bin_edges = np.histogram(im_eq, bins=256, range=(0, 256))
    return [im_eq, hist_orig, hist_eq]

myPic = read_image('bw.jpg', 2)
#myPic = myPic[0:2,0:2,:].round(3)
myPic, hist_orig, hist_eq = histogram_equalize(myPic)
# arrdisplay(myPic, 1)

plt.bar(hist_orig, 'b', hist_eq, 'r')
plt.show()
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

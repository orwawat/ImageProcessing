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
        return rgb.astype(np.float32)


def isGraySacle(im):
    return len(im.shape) >= 3

def getGrayImage(im_float):
    '''
    :param im_float: image astype=np.float32, values are in [0,1]
    :return:
    '''
    if isGraySacle(im_float):
        # Convert to gray scale
        im_gray = color.rgb2gray(im_float)
        im_gray = im_gray.astype(np.float32)
        grayImage = im_gray
    else:
        # Already gray, no need to convert
        grayImage = im_float

    return grayImage

def read_image(filename, representation):
    im = imread(filename)
    # tokenize
    im_float = im.astype(np.float32)
    im_float /= 255

    if representation == 1:
        # Convert to gray
        getGrayImage(im_float)
    else:
        returnImage = im_float

    return returnImage

def imdisplay(filename, representation):
    if representation == 1:
        plt.imshow(read_image(filename, representation), plt.cm.gray)
    else:
        plt.imshow(read_image(filename, representation))
    plt.show()

def rgb2yiq(imRGB):
    return RgbYiqConverter.getYIQ(imRGB)

def yiq2rgb(imYIQ):
    return RgbYiqConverter.getRGB(imYIQ)


def histogram_equalize(im_orig):
    ''' Algorithm:
    1. Given image I(x,y), create a histogram 
    H:
    • For all x,y:  H(I(x,y)) = H(I(x,y)) + 1
    2. Create cumulative histogram S(k):
    • S(0) = H(0);  S(k+1) = S(k) + H(k+1); 
    • Let m be first grey level for which S(m)≠0;
    3. Create Look Up Table (LUT) T(k):
    • T(k) = round{255 × [S(k)‐S(m)] / [S(255)‐S(m)] }
    4. Apply LUT 
    T to image I, get equalized image 
    J
    • J(x,y) = T (I(x,y))
    '''

    if not isGraySacle(im_orig):
        im = yiq2rgb(im_orig)[:, :, 0]
    else:
        im = im_orig

    H = im.histogram()

    return [im_eq, hist_orig, hist_eq]
myPic = read_image('color.jpg', 2)
myPic = myPic#[0:2,0:2,:].round(3)
#print("myPic is: \n")
#print(myPic)
#print("\n\n")

yiqPic = rgb2yiq(myPic)
print("first pic some pixel value: ", myPic[95, 60, :], "\n\n")
print("yiq pic some pixel value: ", yiqPic[95, 60, :], "\n\n")
rgb = yiq2rgb(yiqPic)
print("rgb pic some pixel value: ", rgb[95, 60, :], "\n\n")
#print("********************************")
#print("myPic YIQ pic: \n")
#print(yiqPic)

plt.imshow(rgb)
#plt.imshow(yiqPic[:, :, 0], plt.cm.gray)
plt.show()
#imdisplay('color.jpg', 1)

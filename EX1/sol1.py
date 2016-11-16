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
        redPic = rgbImg[:, :, 0].ravel()
        greenPic = rgbImg[:, :, 1].ravel()
        bluePic = rgbImg[:, :, 2].ravel()
        imgToConvert = np.row_stack((redPic, greenPic, bluePic))
        yiq = np.dot(RgbYiqConverter.yiqToRgb, imgToConvert)
        yiq = yiq.T
        yiq = yiq.reshape((rgbImg.shape[0], rgbImg.shape[1], rgbImg.shape[2]))
        return yiq.astype(np.float32)

    @staticmethod
    def getRGB(yiqImg):
        yPic = yiqImg[:, :, 0].ravel()
        iPic = yiqImg[:, :, 1].ravel()
        qPic = yiqImg[:, :, 2].ravel()
        imgToConvert = np.row_stack((yPic, iPic, qPic))
        rgb = np.dot(np.linalg.inv(RgbYiqConverter.yiqToRgb), imgToConvert)
        rgb = rgb.T
        rgb = rgb.reshape((yiqImg.shape[0], yiqImg.shape[1], yiqImg.shape[2]))
        return rgb.astype(np.float32)

def read_image(filename, representation):
    im = imread(filename)
    im_float = im.astype(np.float32)
    im_float /= 255

    if len(im.shape) >= 3 and representation == 1:
        im_gray = color.rgb2gray(im_float)
        im_gray = im_gray.astype(np.float32)
        returnImage = im_gray
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

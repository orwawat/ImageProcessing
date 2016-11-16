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

        '''
        print("********************************")
        print("imgToConvert YIQ pic: \n")
        print(imgToConvert)

        print("\n\n")
        print("********************************")
        '''

        yiq = np.dot(RgbYiqConverter.yiqToRgb, imgToConvert)
        yiq = yiq.T
        print("\n\n")
        print("********************************")

        print(yiq.shape)
        '''
        print("********************************")
        print("before reshape: \n")
        print(yiq)
        print("\n\n")
        print("********************************")
        '''
        yiq = yiq.reshape((rgbImg.shape[0], rgbImg.shape[1], rgbImg.shape[2]))

        return yiq.astype(np.float32)

        '''
        redY = rgbImg[:,:,0].dot(0.299)
        blueY = rgbImg[:, :, 0].dot(0.587)
        greenY = rgbImg[:, :, 0].dot(0.114)
        Y = redY + blueY + greenY

        redI = rgbImg[:, :, 0].dot(0.596)
        blueI = rgbImg[:, :, 0].dot(-0.275)
        greenI = rgbImg[:, :, 0].dot(-0.321)
        I = redI + blueI + greenI

        redQ = rgbImg[:, :, 0].dot(0.299)
        blueQ = rgbImg[:, :, 0].dot(0.587)
        greenQ = rgbImg[:, :, 0].dot(0.114)
        Q = redQ + blueQ + greenQ

        if (Y == yiq[0, :, :] and I == yiq[:, 0, :] and Q == yiq[:, :, 0]):
            return 1
        else:
            return 0
        '''

    @staticmethod
    def getRGB(yiqImg):
        return np.linalg.inv(RgbYiqConverter.yiqToRgb).dot(yiqImg)

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
myPic = myPic[0:2,0:2,:].round(3)
#print("myPic is: \n")
#print(myPic)
#print("\n\n")

yiqPic = rgb2yiq(myPic)
#print("********************************")
#print("myPic YIQ pic: \n")
#print(yiqPic)
print("\n\n")
plt.imshow(yiqPic)
plt.show()
#imdisplay('color.jpg', 2)
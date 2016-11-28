import numpy as np
from skimage import color
import sol2 as mySol
import matplotlib.pyplot as plt
from scipy.misc import imread as imread

vec = np.array([1,2,3])

def test_DFT():
    print("************* Start test_DFT: \n")
    # Create 1D array
    signal = vec
    # Convert to a 2D array
    correctVal = np.fft.fft(signal)
    signal = signal.reshape((signal.size, 1))
    myVal = mySol.DFT(signal)
    result = np.array_equiv(np.round(myVal(), decimals=3), np.round(correctVal, decimals=3))\
             and type(myVal) == type(correctVal)
    if (result):
        print("VVVVVVVVVVV passed test_DFT VVVVVVVVVVVVVVV\n")
        print("My result is: ", myVal)
        print("Correct Result is: ", correctVal)

    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_DFT")
        print("My result is: ", myVal)
        print("Correct Result is: ", correctVal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def test_IDFT():
    print("************* Start test_IDFT: \n")

    # Create 1D complex array
    fourier_signal = np.array([1 + 2J, 2 + 1J, 3 + 5J, 12 + 12J, 1 + 1J])
    correctVal = np.fft.ifft(fourier_signal)
    fourier_signal = fourier_signal.reshape((fourier_signal.size, 1))

    myVal = mySol.IDFT(fourier_signal)
    result = np.array_equiv(np.round(myVal.flatten(), decimals=3), np.round(correctVal, decimals=3))
    #and type(myVal) == type(correctVal)
    if (result):
        print("VVVVVVVVVVV passed test_IDFT VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_IDFT")
        print("My result is: ", myVal.flatten())
        print("Correct Result is: ", correctVal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def test_DFT_ON_IDFT():
    print("************* Start test_DFT_ON_IDFT: \n")
    # Create 1D array
    signal = vec
    # Convert to a 2D array
    signal = signal.reshape((signal.size, 1))
    myVal = mySol.IDFT(mySol.DFT(signal))
    result = np.array_equiv(myVal, signal) and type(myVal) == type(signal)
    if (result):
        print("VVVVVVVVVVV passed test_DFT_ON_IDFT VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_DFT_ON_IDFT")
        print("My result is: ", myVal)
        print("Starting vec is: ", signal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def test_DFT2():
    print("************* Start test_DFT2: \n")
    # Create 1D array
    img = np.row_stack((vec, vec + 5, vec - 2, vec + 1241)) #, vec + 5, vec - 2
    # Convert to a 2D array
    myVal = mySol.DFT2(img)
    correctVal = np.fft.fft2(img).astype(np.complex128)
    result = np.array_equiv(np.round(myVal, decimals=3), np.round(correctVal, decimals=3))\
             and type(myVal) == type(correctVal)
    if (result):
        print("VVVVVVVVVVV passed test_DFT2 VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_DFT2")
        print("My result is: ", myVal)
        print("Correct Result is: ", correctVal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


def read_gray_im():
    im = imread(".//im.jpg")
    # im = imread(".\\im.jpg")
    # tokenize
    im_float = im.astype(np.float32)
    im_float /= 255
    im_gray = color.rgb2gray(im_float)
    im_gray = im_gray.astype(np.float32)
    grayImage = im_gray
    # plt.imshow(im_gray, plt.cm.gray)
    # plt.show()
    return grayImage

def test_conv_der():
    im = read_gray_im()
    der_im = mySol.conv_der(im)
    plt.imshow(der_im, plt.cm.gray)
    # plt.show()

def test_fourier_der():
    im = read_gray_im()
    der_im = mySol.fourier_der(im)
    # plt.imshow(der_im, plt.cm.gray)

# test_IDFT()
# test_DFT()
# test_DFT_ON_IDFT()
# test_DFT2()
# test_conv_der()
test_fourier_der()

a = 76

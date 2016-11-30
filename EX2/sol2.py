import numpy as np
from scipy import signal as sig, linalg
from scipy.misc import imread as imread
from skimage import color

DERIVATIVE_KERNEL = np.array([1, 0, -1])


def read_image(filename, representation):
    im = imread(filename)
    # tokenize
    im_float = im.astype(np.float32)
    im_float /= 255

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

    if representation == 1:
        # Convert to gray
        returnImage = getGrayImage(im_float)
    else:
        returnImage = im_float

    return returnImage

def create_dft_matrix(size):
    u, x = np.meshgrid(np.arange(size), np.arange(size))
    omega = np.exp(-2J * np.pi / size)
    return np.power(omega, u * x)

def DFT(signal):
    N = signal.shape[0]
    dft_matrix = create_dft_matrix(N)
    fourier_matrix = np.dot(dft_matrix, signal)

    M = signal.shape[1]
    dft_matrix = create_dft_matrix(M)
    fourier_matrix = np.dot(dft_matrix, fourier_matrix.T).T
    return fourier_matrix

def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    dft_matrix = linalg.inv(create_dft_matrix(N))
    fourier_matrix = np.dot(dft_matrix, fourier_signal)

    M = fourier_signal.shape[1]
    dft_matrix = linalg.inv(create_dft_matrix(M))
    fourier_matrix = np.dot(dft_matrix, fourier_matrix.T).T
    return fourier_matrix

def DFT2(image):
    return DFT(image)

def IDFT2(fourier_image):
    return IDFT(fourier_image)


def conv_der(im):
    x_kerenl = np.array(DERIVATIVE_KERNEL).reshape(1, DERIVATIVE_KERNEL.size)
    y_kerenl = x_kerenl.T

    dx = sig.convolve2d(im, x_kerenl, mode='same')
    dy = sig.convolve2d(im, y_kerenl, mode='same')

    dx = dx.reshape(im.shape)
    dy = dy.reshape(im.shape)
    magntiatude = np.sqrt(np.square(np.abs(dx)) + np.square(np.abs(dy)))

    return magntiatude.astype(np.float32)


def fourier_der1(im):
    fourier_im = DFT2(im)
    fourier_im = np.fft.fftshift(fourier_im)

    u = np.linspace((np.ceil(im.shape[1] / -2)), np.floor((im.shape[1] / 2)),
                    num=im.shape[1], endpoint=False, dtype=np.int)
    uMatrix, ua = np.meshgrid(u, np.arange(im.shape[0]))

    v = np.linspace((np.ceil(im.shape[0] / -2)), np.floor((im.shape[0] / 2)),
                    num=im.shape[0], endpoint=False, dtype=np.int)
    vMatrix, va = np.array(np.meshgrid(v, np.arange(im.shape[1])))#.reshape((2, im.shape[1]))

    dx = IDFT2(fourier_im * uMatrix * ua).real.astype(np.float32)

    dy = IDFT2(fourier_im * (vMatrix * va).T).real.astype(np.float32)
    magntiatude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).real.astype(np.float32)
    # imsave(os.getcwd() + '/fourier_x_derivative.jpg', dx)
    # imsave(os.getcwd() + '/fourier_y_derivative.jpg', dy)
    # imsave(os.getcwd() + '/fourier_magnitude1.jpg', magntiatude)
    # imsave(os.getcwd() + '/fourier_magnitude2.jpg', magntiatude[:,:,1])
    return magntiatude

def fourier_der(im):
    f_im = DFT2(im)
    # f_im = np.fft.fft2(im)
    f_im_centered = np.fft.fftshift(f_im)

    u = np.arange(np.ceil(im.shape[0] / -2), np.ceil(im.shape[0] / 2)).reshape(im.shape[0], 1)#[:, np.newaxis]#
    dx = (f_im_centered * u)
    dx = IDFT2(np.fft.ifftshift(dx))#.real.astype(np.float32)
    # dx  = IDFT2(dx)

    v = np.arange(np.ceil(im.shape[1] / -2), np.floor(im.shape[1] / 2))#[:, np.newaxis]#.reshape(1, im.shape[1])
    dy = f_im_centered * v
    dy = IDFT2(np.fft.ifftshift(dy))#.real.astype(np.float32)
    # dy = IDFT2(dy)
    # dx = dx * 2J * np.pi / im.shape[0]
    # dy = dy * 2J * np.pi / im.shape[1]
    magntiatude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)#.real.astype(np.float32)
    # imsave(os.getcwd() + '/fourier_x_derivative.jpg', dx)
    # imsave(os.getcwd() + '/fourier_y_derivative.jpg', dy)
    # imsave(os.getcwd() + '/fourier_magnitude.jpg', magntiatude)
    return magntiatude

def create_kernel(size):
    kernel = base_kernel = np.array([[1, 1]])

    for i in range(size - 2):
        kernel = sig.convolve2d(base_kernel, kernel)
    kernel = sig.convolve2d(kernel, kernel.T)

    total_kernel = np.sum(kernel)
    # kernel = kernel.astype(np.float32) / total_kernel
    return kernel

def blur_spatial(im, kernel_size):
    kernel = create_kernel(kernel_size)
    blur_im = sig.convolve2d(im, kernel, mode='same', boundary='wrap').astype(np.float32)
    return blur_im

def blur_fourier(im, kernel_size):
    fourier_im = DFT2(im)

    kernel = np.zeros(shape=im.shape)
    kernel_location = (np.floor(im.shape[0] / 2) - np.floor(kernel_size / 2),
                       np.floor(im.shape[1] / 2 - np.floor(kernel_size / 2)))
    kernel[kernel_location[0]: kernel_location[0] + kernel_size, kernel_location[1]: kernel_location[1] + kernel_size] \
        = create_kernel(kernel_size)

    kernel = np.fft.ifftshift(kernel)
    fourier_kernel = DFT2(kernel)

    fourier_blur = fourier_im * fourier_kernel
    blur_im = IDFT2(fourier_blur)
    return blur_im.real.astype(np.float32)


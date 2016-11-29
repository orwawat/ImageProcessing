import numpy as np
import os
from scipy import signal, linalg
from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt

DERIVATIVE_KERNEL = np.array([1, 0, -1])


def DFT1(signal):
    N = signal.shape[0]
    dft_matrix = linalg.dft(N)
    return np.dot(dft_matrix, signal)

def DFT(signal):
    N = signal.shape[0]
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    curExp = np.exp(-2 * np.pi * 1J * u * x / N)
    result_vec = np.dot(signal, curExp).astype(np.complex128)
    result_vec = np.row_stack(result_vec)
    # print(result_vec.shape)
    return result_vec

def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    curExp = np.exp(2 * np.pi * 1J * x * u / N)
    # TODO: http://moodle2.cs.huji.ac.il/nu16/mod/forum/discuss.php?d=6956
    result_vec = (np.dot(fourier_signal.flatten(), curExp) / N).real.astype(np.float32)
    return np.row_stack(result_vec)

def DFT2(image):
    return np.fft.fft2(image)
    row_fourier_image = fourier_image = np.zeros(image.shape, dtype=np.complex128)
    # N = image.shape[0]
    # u, x = np.meshgrid(np.arange(N), np.arange(N))
    # for v in range(image.shape[1]):
    #     curExp = np.exp(-2 * np.pi * 1J * u * x / N)
    #     fourier_image[:, v] = np.dot(curExp, DFT(image[:, v]).flatten()) / N

    # Compute 1D fourier on each row
    for i in range(image.shape[0]):
        row_fourier_image[i, :] = DFT(image[i, :]).reshape(image.shape[1])

    # Compute the 1D fourier on each column
    for j in range(image.shape[1]):
        fourier_image[:, j] = DFT(row_fourier_image[:, j]).reshape(1, image.shape[0])


    return fourier_image


def IDFT2(fourier_image):
    return np.fft.ifft2(fourier_image)



def conv_der(im):
    x_kerenl = np.row_stack((np.zeros(DERIVATIVE_KERNEL.shape), DERIVATIVE_KERNEL, np.zeros(DERIVATIVE_KERNEL.shape)))
    y_kerenl = np.column_stack((np.zeros(DERIVATIVE_KERNEL.shape),
                                np.reshape(DERIVATIVE_KERNEL, (DERIVATIVE_KERNEL.shape[0], 1)),
                                np.zeros(DERIVATIVE_KERNEL.shape)))

    dx = signal.convolve2d(im, x_kerenl, mode='same')
    dy = signal.convolve2d(im, y_kerenl, mode='same')

    dx = dx.reshape(im.shape)
    dy = dy.reshape(im.shape)
    magntiatude = np.sqrt(np.square(dx) + np.square(dy))

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
    fourier_im = DFT2(im)
    fourier_im = np.fft.fftshift(fourier_im)

    u = np.linspace((np.ceil(im.shape[0] / -2)), np.floor((im.shape[0] / 2)),
                    num=im.shape[0], endpoint=False, dtype=np.int)

    dx = (fourier_im.T * u).T
    # dx = np.fft.fftshift(fourier_dx)
    dx = IDFT2(dx).real.astype(np.float32)
    
    v = np.linspace((np.ceil(im.shape[1] / -2)), np.floor((im.shape[1] / 2)),
                    num=im.shape[1], endpoint=False, dtype=np.int)
    v = np.reshape(v, (1, v.shape[0]))
    dy = fourier_im * v
    # dy = np.fft.fftshift(fourier_dy)
    dy = IDFT2(dy).real.astype(np.float32)
    # show_plot(np.log(1 + np.abs(dx)))
    # show_plot(np.log(1+np.abs(dy)))
    magntiatude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).real.astype(np.float32)
    # imsave(os.getcwd() + '/fourier_x_derivative.jpg', dx)
    # imsave(os.getcwd() + '/fourier_y_derivative.jpg', dy)
    # imsave(os.getcwd() + '/fourier_magnitude.jpg', magntiatude)
    return magntiatude

def create_kernel(size):
    kernel = base_kernel = np.array([[1, 1]])

    for i in range(size - 2):
        kernel = signal.convolve2d(base_kernel, kernel)
    kernel = signal.convolve2d(kernel, kernel.T)

    total_kernel = np.sum(kernel)
    kernel = kernel.astype(np.float32) / total_kernel
    return kernel

def blur_spatial(im, kernel_size):
    kernel = create_kernel(kernel_size)
    blur_im = signal.convolve2d(im, kernel, mode='same', boundary='wrap').astype(np.float32)
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

def show_plot(s_im):
    plt.imshow(s_im, plt.cm.gray)
    plt.show()

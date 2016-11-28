import numpy as np
import os
from scipy import signal
from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt

DERIVATIVE_KERNEL = np.array([1, 0, -1])

def DFT1(signal):
    N = signal.shape[0]
    M = signal.shape[1]
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    v, y = np.meshgrid(np.arange(M), np.arange(M))
    curExp = np.exp(-2J * np.pi * ((np.dot(u, x) / N) + (np.dot(v, y) / M)))
    result_vec = np.dot(signal, curExp).astype(np.complex128)
    return np.row_stack(result_vec)



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
    # return np.fft.fft2(image).real.astype(np.float32)
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
    return np.fft.ifft2(fourier_image).real.astype(np.float32)



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

    show_plot(dx)
    show_plot(dy)
    show_plot(magntiatude)
    # imsave('C:\\Users\\Alon\\Documents\\HUJI\\ImageProcessing\\Exercises\\EX2\\x_derivative.jpg', dx)
    # imsave('C:\\Users\\Alon\\Documents\\HUJI\\ImageProcessing\\Exercises\\EX2\\y_derivative.jpg', dy)
    # imsave('C:\\Users\\Alon\\Documents\\HUJI\\ImageProcessing\\Exercises\\EX2\\magnitude.jpg', magntiatude)
    return magntiatude.astype(np.float32)

def fourier_der(im):
    fourier_im = DFT2(im)
    fourier_im = np.fft.fftshift(fourier_im)

    u = np.linspace((im.shape[0] / -2), (im.shape[0] / 2), num=im.shape[0], endpoint=False, dtype=np.int)
    fourier_dx = (fourier_im.T * u).T
    fourier_dx = np.fft.ifftshift(fourier_dx)
    dx = IDFT2(fourier_dx.real).astype(np.float32)
    
    v = np.linspace((im.shape[1] / -2), (im.shape[1] / 2), num=im.shape[1], endpoint=False, dtype=np.int)
    fourier_dy = fourier_im * v
    fourier_dy = np.fft.ifftshift(fourier_dy)
    dy = IDFT2(fourier_dy.real).astype(np.float32)

    magntiatude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)

    imsave(os.getcwd() + '/fourier_x_derivative.jpg', dx)
    imsave(os.getcwd() + '/fourier_y_derivative.jpg', dy)
    imsave(os.getcwd() + '/fourier_magnitude.jpg', magntiatude)
'''
def fourier_der_loop(im):
    fourier_im = DFT2(im)
    # u_vals = np.linspace(0, im.shape[0], im.shape[0], endpoint=False)
    # u_vals = np.reshape(u_vals, (1, u_vals.shape[0]))
    # fourier_dx = u_vals * fourier_im
    fourier_dx = np.zeros(fourier_im.shape)
    for i in range(im.shape[0]):
        fourier_dx[i, :] = fourier_im[i, :] * i
    dx = IDFT2(fourier_dx.real).astype(np.float32)

    v_vals = np.linspace(0, im.shape[1], im.shape[1])
    v_vals = np.reshape(v_vals, (1, v_vals.shape[0]))
    fourier_dy = fourier_im * v_vals
    dy = IDFT2(fourier_dy.real).astype(np.float32)

    magntiatude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

    imsave(os.getcwd() + '\\fourier_x_derivative.jpg', dx)
    imsave(os.getcwd() + '\\fourier_y_derivative.jpg', dy)
    imsave(os.getcwd() + '\\fourier_magnitude.jpg', magntiatude)
    show_plot(dx)
    show_plot(dy)
    show_plot(magntiatude)
    return magntiatude.astype(np.float32)
'''
def show_plot(s_im):
    plt.imshow(s_im, plt.cm.gray)
    plt.show()

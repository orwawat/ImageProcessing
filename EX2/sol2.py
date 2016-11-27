import numpy as np
from scipy.misc import imread as imread, imsave as imsave




def DFT(signal):
    N = signal.shape[0]
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    curExp = np.exp(-2 * np.pi * 1J * u * x / N)
    return np.dot(signal, curExp).astype(np.complex128)


def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    curExp = np.exp(2 * np.pi * 1J * x * u/ N)
    return (fourier_signal * curExp / N).astype(np.complex128)

def DFT2(image):
    return

def IDFT2(fourier_image):
    return
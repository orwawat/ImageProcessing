import numpy as np
from scipy.misc import imread as imread, imsave as imsave




def DFT(signal):
    N = signal[0].size
    u = np.arange(N)
    curExp = np.exp(-2 * np.pi * 1J * u / N)
    return (signal * curExp).astype(np.complex128)


def IDFT(fourier_signal):
    N = fourier_signal[0].size
    x = np.arange(N)
    curExp = np.exp(2 * np.pi * 1J * x / N)
    return (fourier_signal * curExp / N).astype(np.complex128)

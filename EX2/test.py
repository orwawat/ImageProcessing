import numpy as np
import sol2 as mySol

def get_random_vec():
    signal = np.random.random(np.random.randint(1, 100))
    signal = signal * 14
    signal = signal.astype(np.float32)
    return signal

def get_random_complex_vec():
    vec_size = np.random.randint(1, 100)
    fourier_signal = np.random.rand(vec_size) + np.random.rand(vec_size) * 1j
    return fourier_signal.astype(np.complex128)

def test_DFT():
    print("************* Start test_DFT: \n")
    # Create 1D array
    signal = get_random_vec()
    signal = np.array([1,2,3,4])
    # Convert to a 2D array
    # signal = signal.reshape((signal.size, 1))
    myVal = mySol.DFT(signal)
    correctVal = np.fft.fft(signal)
    result = np.array_equiv(myVal, correctVal) and type(myVal) == type(correctVal)
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
    fourier_signal = get_random_complex_vec()
    fourier_signal = fourier_signal.reshape((fourier_signal.size, 1))

    myVal = mySol.IDFT(fourier_signal)
    correctVal = np.fft.ifft(fourier_signal)
    result = np.array_equiv(myVal, correctVal) and type(myVal) == type(correctVal)
    if (result):
        print("VVVVVVVVVVV passed test_IDFT VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_IDFT")
        print("My result is: ", myVal)
        print("Correct Result is: ", correctVal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def test_DFT_ON_IDFT():
    print("************* Start test_DFT_ON_IDFT: \n")
    # Create 1D array
    signal = get_random_vec()
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
    img = np.random.rand(np.random.randint(100, 500), np.random.randint(100, 500))
    # Convert to a 2D array
    myVal = mySol.DFT2(signal)
    correctVal = np.fft.fft2(signal)
    result = np.array_equiv(myVal, correctVal) and type(myVal) == type(correctVal)
    if (result):
        print("VVVVVVVVVVV passed test_DFT2 VVVVVVVVVVVVVVV\n")
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Faild test_DFT")
        print("My result is: ", myVal)
        print("Correct Result is: ", correctVal)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# test_IDFT()
test_DFT()
# test_DFT_ON_IDFT()
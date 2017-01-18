import numpy as np
from skimage import color
from scipy.misc import imread
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def read_image(filename, representation):
    im = imread(filename)
    # tokenize
    if (np.amax(im) > 1):
        im_float = im.astype(np.float32)
        im_float /= 255
    else:
        im_float = im.astype(np.float32)

    if representation == 1:
        # Convert to gray
        if len(im.shape) < 3:
            # Already gray, no need to convert
            returnImage = im_float
        else:
            # Convert to gray scale
            im_gray = color.rgb2gray(im_float)
            im_gray = im_gray  # .astype(np.float32)
            returnImage = im_gray
    else:
        returnImage = im_float

    return returnImage  # .astype(np.float32)

def calculate_crop_start_location(im_shape, crop_size):
    width_start = np.random.randint(im_shape[0] - crop_size[0], size=1)
    height_start = np.random.randint(im_shape[1] - crop_size[1], size=1)
    return width_start, height_start

def get_patch(im, start_location, crop_size):
    return im[start_location[0]: start_location[0] + crop_size[0], \
           start_location[1] : start_location[1] + crop_size[1]]

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    image_cache = {}
    while True:
        source_batch = np.empty(shape=(batch_size,))
        target_batch = np.empty(shape=(batch_size,))
        for i in range(batch_size):
            im = None
            fileName = filenames[np.random.randint(len(filenames), size=1)]
            if fileName in image_cache:
                im = image_cache.get(fileName)
            else:
                im = read_image(fileName, 1)
                image_cache.update(fileName, im)

            corrupted_im = corruption_func(im)
            start_loc = calculate_crop_start_location(im.shape, crop_size)
            source_batch[i] = get_patch(im, start_loc, crop_size)
            target_batch[i] = get_patch(corrupted_im, start_loc, crop_size)
        yield (source_batch, target_batch)






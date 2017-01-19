import numpy as np
from skimage import color
from scipy.misc import imread
from keras.layers import Convolution2D, Activation, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from sol5_utils import *
from scipy.ndimage.filters import convolve
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
            im_gray = im_gray
            returnImage = im_gray
    else:
        returnImage = im_float

    return returnImage.astype(np.float32)


def calculate_crop_start_location(im_shape, crop_size):
    width_start = np.random.randint(im_shape[0] - crop_size[0], size=1)[0]
    height_start = np.random.randint(im_shape[1] - crop_size[1], size=1)[0]
    return width_start, height_start


def get_patch(im, start_location, crop_size):
    return im[start_location[0]: start_location[0] + crop_size[0],
              start_location[1]: start_location[1] + crop_size[1]]

image_cache = {}


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    while True:
        source_batch = np.empty(shape=(batch_size,1,crop_size[0], crop_size[1]), dtype=np.float32)
        target_batch = np.empty(shape=(batch_size,1,crop_size[0], crop_size[1]), dtype=np.float32)
        for i in range(batch_size):
            im = None
            fileName = np.random.choice(filenames)
            if fileName in image_cache:
                im = image_cache[fileName]
            else:
                im = read_image(fileName, 1)
                image_cache[fileName] = im

            corrupted_im = corruption_func(im)
            start_loc = calculate_crop_start_location(im.shape, crop_size)
            source_batch[i,:,:,:] = (get_patch(corrupted_im, start_loc, crop_size) - 0.5)
            target_batch[i,:,:,:] = (get_patch(im, start_loc, crop_size) - 0.5)
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    output_tensor = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Convolution2D(num_channels, 3, 3, border_mode='same')(output_tensor)
    output_tensor = merge([input_tensor, output_tensor], mode='sum')
    return output_tensor


def build_nn_model(height, width, num_channels):
    num_resdiual_block = 5
    input_model = Input(shape=(1,height,width))
    init_block = Convolution2D(num_channels, 3, 3, border_mode='same')(input_model)
    init_block = Activation('relu')(init_block)
    # first_addition_layer = out_model
    out_model = resblock(init_block, num_channels)

    for i in range(num_resdiual_block - 1):
        out_model = resblock(out_model, num_channels)

    out_model = merge([init_block, out_model], mode='sum')
    out_model = Convolution2D(1, 3,3, border_mode='same')(out_model)

    model = Model(input=input_model, output=out_model)

    return model


def train_model(model, images, corruption_func, batch_size, samples_per_epoch,
                num_epochs, num_valid_samples):
    crop_size = (model.input_shape[-2], model.input_shape[-1])
    start_test_location = int(len(images) * 0.8)

    np.random.shuffle(images)

    traning_set = load_dataset(images[:start_test_location], batch_size,
                               corruption_func, crop_size)
    validation_set = load_dataset(images[start_test_location:], batch_size,
                                  corruption_func, crop_size)
    adam_opt = Adam(beta_2=0.9)
    model.compile(loss='mean_squared_error', optimizer=adam_opt)
    model.fit_generator(traning_set,
                        samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=validation_set, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model, num_channels):
    corrupted_image -= 0.5
    model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    model.set_weights(base_model.get_weights())
    restored_image = model.predict(corrupted_image[np.newaxis,np.newaxis,...], 1)
    restored_image += 0.5
    return restored_image.clip(0,1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    sigma = np.random.uniform(min_sigma, max_sigma, 1)
    corrupted_im = image + np.random.normal(0, sigma, image.shape)
    return corrupted_im


def gaussian_noise_corruption(im):
    min_sigma = 0
    max_sigma = 0.2
    return add_gaussian_noise(im, min_sigma, max_sigma)




def learn_denoising_model(quick_mode=False):
    im_paths = images_for_denoising()
    patch_size = (24, 24)
    num_channels = 48
    corruption_func = gaussian_noise_corruption
    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30
    else:
        batch_size = 100
        samples_per_epoch = 10000
        num_epochs = 5
        num_valid_samples = 1000

    model = build_nn_model(patch_size[0], patch_size[1], num_channels)
    train_model(model, im_paths, corruption_func, batch_size, samples_per_epoch,
                num_epochs, num_valid_samples)

    return model, num_channels



def add_motion_blur(image, kernel_size, angle):
    corrupted = convolve(image, motion_blur_kernel(kernel_size, angle))
    return corrupted

def random_motion_blur(image, list_of_kernel_sizes):
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(quick_mode=False):
    im_paths = images_for_deblurring()
    patch_size = (16, 16)
    num_channels = 32
    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30
    else:
        batch_size = 100
        samples_per_epoch = 10000
        num_epochs = 10
        num_valid_samples = 1000

    model = build_nn_model(patch_size[0], patch_size[1], num_channels)
    train_model(model, im_paths, lambda im: random_motion_blur(im, [7]), batch_size,
                samples_per_epoch, num_epochs, num_valid_samples)

    return model, num_channels
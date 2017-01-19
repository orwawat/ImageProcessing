import numpy as np
from scipy import signal as sig
from scipy.misc import imread, imsave
from skimage import color
from timeit import timeit
import matplotlib.pyplot as plt
import sol5 as mySol
from sol5_utils import *

from keras.models import model_from_json

# IM_NAMES = ['lena.jpg', 'jerusalem.jpg', 'Low Contrast.jpg', 'monkey.jpg']
IM_NAMES = ['text1.jpg', 'text2.jpg','text3.jpg']



def get_images():
    res = []
    for im in IM_NAMES:
        res.append(get_specific_image(im))
    return res


# def get_image():
#     return get_specific_image(IM_NAME)

def get_specific_image(name):
    im = imread(".//my_images//" + name)
    im = color.rgb2gray(im)
    im_gray = im.astype(np.float32)
    # plt.imshow(im_gray, plt.cm.gray)# plt.show()
    return im_gray.astype(np.float32)


def show_plot(s_im):
    plt.imshow(s_im, plt.cm.gray)
    plt.show()


def show_3_im(im1, im2, im3):
    plt.subplot(2, 2, 1)
    plt.imshow(im1, plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.imshow(im2, plt.cm.gray)
    plt.subplot(2, 2, 3)
    plt.imshow(im3, plt.cm.gray)
    plt.show()

def test_add_gaussian_noise():
    im = get_images()[0]
    corupted = mySol.add_gaussian_noise(im, 0, 1)
    show_3_im(im, corupted, [])


def test_load_dataset():
    fileNames = images_for_denoising()
    crop_size = (20,20)
    batch_size = 1
    data_generator = mySol.load_dataset(fileNames, batch_size,lambda x: x, crop_size)
    next(data_generator)
    next(data_generator)
    next(data_generator)
    print(next(data_generator))


def test_learn_deblurring_model():
    ims = get_images()
    model, channels = mySol.learn_deblurring_model(False)

    model_json = model.to_json()
    with open("model_deblurring5.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_deblurring5.h5")

    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # #load weights into new model
    # loaded_model.load_weights("model.h5")

    for i in range(len(ims)):
        im_corrupte = mySol.random_motion_blur(ims[i], [7])
        restored_im = mySol.restore_image(im_corrupte, model, channels)[0,0,...]
        # imsave(os.path.join(os.getcwd(), dir, 'restored_im' + '_lena.jpg'), restored_im)
        # imsave(os.path.join(os.getcwd(), dir, 'im_corrupte' + '_lena.jpg'), im_corrupte)
        show_3_im(im_corrupte, ims[i], restored_im)

def test_learn_denoising_model():
    ims = get_images()
    model, channels = mySol.learn_denoising_model(False)

    model_json = model.to_json()
    with open("model5.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model5.h5")

    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # #load weights into new model
    # loaded_model.load_weights("model.h5")

    for i in range(len(ims)):
        im_corrupte = mySol.add_gaussian_noise(ims[i], 0.0, 0.2)
        restored_im = mySol.restore_image(im_corrupte, model, channels)[0, 0, ...]
        # imsave(os.path.join(os.getcwd(), dir, 'restored_im' + '_lena.jpg'), restored_im)
        # imsave(os.path.join(os.getcwd(), dir, 'im_corrupte' + '_lena.jpg'), im_corrupte)
        show_3_im(im_corrupte, ims[i], restored_im)



# arr = np.arange(100)
# print(np.random.choice(arr))
# print(np.random.randint(0, len(arr), 1)[0])
# test_load_dataset()
# test_add_gaussian_noise()
# test_learn_denoising_model()
test_learn_deblurring_model()
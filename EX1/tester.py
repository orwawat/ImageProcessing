from sol1 import *
import numpy as np
import scipy.misc
import os.path
import filecmp

images = np.array(["jerusalem.jpg", "Low Contrast.jpg", "monkey.jpg"])

for img in images:
    # open the image as grayscale and save results

    im = read_image(os.path.join("tester_files\\", img), 1)
    im_quantized, error = quantize(im, 7, 100)
    im_equalized, hist_orig, hist_equalized = histogram_equalize(im)

    # # save results - for my result only (the results to compare with)
    # scipy.misc.imsave(os.path.join("tester_files\\compare_files\\grayscale\\quantized\\", img), im_quantized)
    # scipy.misc.imsave(os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img), im_equalized)
    # np.savetxt(os.path.join("tester_files\\compare_files\\grayscale\\quantized\\", img + "_error.txt"), error)
    # np.savetxt(os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img + "_hist_orig.txt"), hist_orig)
    # np.savetxt(os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img + "_hist_equalized.txt"), hist_equalized)

    # save results
    scipy.misc.imsave(os.path.join("tester_files\\output_files\\grayscale\\quantized\\", img), im_quantized)
    scipy.misc.imsave(os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img), im_equalized)
    np.savetxt(os.path.join("tester_files\\output_files\\grayscale\\quantized\\", img + "_error.txt"), error)
    np.savetxt(os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img + "_hist_orig.txt"), hist_orig)
    np.savetxt(os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img + "_hist_equalized.txt"), hist_equalized)

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\grayscale\\quantized\\", img),
        os.path.join("tester_files\\output_files\\grayscale\\quantized\\", img),
        shallow=False
    ):
        print("PASSED :)\tquantized " + img)
    else:
        print("FAILED :(\tquantized " + img)

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img),
        os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img)
    else:
        print("FAILED :(\tequalized " + img)

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\grayscale\\quantized\\", img + "_error.txt"),
        os.path.join("tester_files\\output_files\\grayscale\\quantized\\", img + "_error.txt"),
        shallow=False
    ):
        print("PASSED :)\tquantized " + img + " error output")
    else:
        print("FAILED :(\tquantized " + img + " error output")

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img + "_hist_orig.txt"),
        os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img + "_hist_orig.txt"),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img + " histogram original output")
    else:
        print("FAILED :(\tequalized " + img + " histogram original output")

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\grayscale\\equalized\\", img + "_hist_equalized.txt"),
        os.path.join("tester_files\\output_files\\grayscale\\equalized\\", img + "_hist_equalized.txt"),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img + " histogram equalized output")
    else:
        print("FAILED :(\tequalized " + img + " histogram equalized output")

for img in images:
    # open the image as RGB and save results
    im = read_image(os.path.join("tester_files\\", img), 2)
    im_quantized, error = quantize(im, 7, 100)
    im_equalized, hist_orig, hist_equalized = histogram_equalize(im)

    # # save results - for my result only (the results to compare with)
    # scipy.misc.imsave(os.path.join("tester_files\\compare_files\\rgb\\quantized\\", img), im_quantized)
    # scipy.misc.imsave(os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img), im_equalized)
    # np.savetxt(os.path.join("tester_files\\compare_files\\rgb\\quantized\\", img + "_error.txt"), error)
    # np.savetxt(os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img + "_hist_orig.txt"), hist_orig)
    # np.savetxt(os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img + "_hist_equalized.txt"), hist_equalized)

    # save results
    scipy.misc.imsave(os.path.join("tester_files\\output_files\\rgb\\quantized\\", img), im_quantized)
    scipy.misc.imsave(os.path.join("tester_files\\output_files\\rgb\\equalized\\", img), im_equalized)
    np.savetxt(os.path.join("tester_files\\output_files\\rgb\\quantized\\", img + "_error.txt"), error)
    np.savetxt(os.path.join("tester_files\\output_files\\rgb\\equalized\\", img + "_hist_orig.txt"), hist_orig)
    np.savetxt(os.path.join("tester_files\\output_files\\rgb\\equalized\\", img + "_hist_equalized.txt"), hist_equalized)

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\rgb\\quantized\\", img),
        os.path.join("tester_files\\output_files\\rgb\\quantized\\", img),
        shallow=False
    ):
        print("PASSED :)\tquantized " + img)
    else:
        print("FAILED :(\tquantized " + img)

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img),
        os.path.join("tester_files\\output_files\\rgb\\equalized\\", img),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img)
    else:
        print("FAILED :(\tequalized " + img)

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\rgb\\quantized\\", img + "_error.txt"),
        os.path.join("tester_files\\output_files\\rgb\\quantized\\", img + "_error.txt"),
        shallow=False
    ):
        print("PASSED :)\tquantized " + img + " error output")
    else:
        print("FAILED :(\tquantized " + img + " error output")

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img + "_hist_orig.txt"),
        os.path.join("tester_files\\output_files\\rgb\\equalized\\", img + "_hist_orig.txt"),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img + " histogram original output")
    else:
        print("FAILED :(\tequalized " + img + " histogram original output")

    print("-------------------------------------------------------------------------------")

    if filecmp.cmp(
        os.path.join("tester_files\\compare_files\\rgb\\equalized\\", img + "_hist_equalized.txt"),
        os.path.join("tester_files\\output_files\\rgb\\equalized\\", img + "_hist_equalized.txt"),
        shallow=False
    ):
        print("PASSED :)\tequalized " + img + " histogram equalized output")
    else:
        print("FAILED :(\tequalized " + img + " histogram equalized output")

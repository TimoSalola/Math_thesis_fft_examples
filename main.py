import numpy
from numpy import array
from PIL import Image

'''
This file contains examples on how periodical signal of known period can be removed from 1d and 2d signals
Comment out functions on lines 170 onwards to disable/enable a test function
'''


# 1d signal separation example from section 4.3 in the thesis
def test_fft_removal_on_chapter_4_3_array():
    test_signal = [1, 4, 4, -9, -1, 3, 7, -4, -6, 1, 6, -1, -1, -4, 4, -2, 2, 1, -1, -4]
    # the test signal is the same as used in section 4.3
    # test signal is the sum of [-1,1,4,-4] and [2,3,0,-5,0] repeated until 20 values

    filtered_signal = fft_remove_every_nth_term(test_signal, 5)
    # removing terms 4, 8, 12 and 16 from the signal

    projected_signal = cast_complex_array_to_integers(filtered_signal)
    # casting values to integers between -255 and 255 as filtered signal will contain non-integer values and complex
    # numbers due to floating point errors

    print("Test signal: " + str(test_signal))
    print("After fourier nth term removal: " + str(filtered_signal))
    print("After projection: " + str(projected_signal))

    print("5 first terms: " + str(projected_signal[0:5]))


# 2d signal error removal on flat background, not shown or mentioned in the thesis
def test_fft_removal_on_pattern():
    # testing error removal on grey image containing only the error pattern

    img = Image.open("testimages/pattern_only.jpg")
    # image is 800*600px with an error pattern of 20*20px. 40 vertical repeats x 30 horizontal repeats

    img_array = array(img)
    # transforming image into a numpy array

    perform_fft_multiples_removal_on_rgb_img_2way(img_array, 40, 30)
    # performing horizontal and vertical fft term removal for a signal of 40 and 30 repeats

    array_to_img(img_array)
    # displaying the result


# 2d signal error removal on flat background with text, not shown or mentioned in the thesis
def test_fft_removal_on_pattern_text():
    # testing error removal on grey image containing only the error pattern

    img = Image.open("testimages/pattern_text.jpg")
    # image is 800*600px with an error pattern of 20*20px. 40 vertical repeats x 30 horizontal repeats

    img_array = array(img)
    # transforming image into a numpy array

    perform_fft_multiples_removal_on_rgb_img_2way(img_array, 40, 30)
    # performing horizontal and vertical fft term removal for a signal of 40 and 30 repeats

    array_to_img(img_array)
    # displaying the result


# 2d signal error removal on a test image, used in thesis section 4.4
def test_fft_removal_on_pattern_img():
    # testing error removal on grey image containing only the error pattern

    img = Image.open("testimages/pattern_text.jpg")
    # image is 800*600px with an error pattern of 20*20px. 40 vertical repeats x 30 horizontal repeats

    img_array = array(img)
    # transforming image into a numpy array

    perform_fft_multiples_removal_on_rgb_img_2way(img_array, 40, 30)
    # performing horizontal and vertical fft term removal for a signal of 40 and 30 repeats

    array_to_img(img_array)
    # displaying the result


# removing every n:th term multiples from a signal
def fft_remove_every_nth_term(signal, n):
    signal_fft = numpy.fft.fft(signal)
    # calculating fft terms with numpy

    for x in range(len(signal_fft)):
        if x % n == 0 and x != 0:
            # if term is a non zero multiple of n
            signal_fft[x] = 0

    signal_restored = numpy.fft.ifft(signal_fft)
    # reversing fft with numpy inverse fft

    return signal_restored


def perform_fft_multiples_removal_on_rgb_img_2way(img_array, vertical, horizontal):
    perform_fft_multiples_removal_on_rgb_img(img_array, vertical)
    # removing vertical signal with period of -vertical-

    img_array = numpy.rot90(img_array)
    # rotating image, this means that the same function can be used again

    perform_fft_multiples_removal_on_rgb_img(img_array, horizontal)
    # removing horizontal signal with period of -horizontal-


def perform_fft_multiples_removal_on_rgb_img(img_array, multiples):
    # reading image depth dimension
    img_depth = img_array.shape[2]

    # for each layer in the image
    for d in range(img_depth):
        print("Removing signals on layer: " + str(d))
        image_layer = img_array[:, :, d]

        img_array[:, :, d] = perform_fft_multiples_removal_on_single_layer(image_layer, multiples)
        # returning layer to matrix


# this function performs fft ter mmultiple removal on a 2d matrix
def perform_fft_multiples_removal_on_single_layer(layer, multiples):
    for y in range(len(layer)):
        line = layer[y]
        filtered = perform_fft_multiples_removal_on_single_line(line, multiples)
        layer[y] = cast_complex_array_to_integers_8bit(filtered)

    return layer


# this function performs fft term multiple removal on a 1d array
def perform_fft_multiples_removal_on_single_line(line, multiples):
    filtered_signal = fft_remove_every_nth_term(line, multiples)
    return filtered_signal


# this function projects a discrete complex signal to integer domain
def cast_complex_array_to_integers(signal):
    output = []
    for x in range(len(signal)):
        value = int(signal[x].real)

        output.append(value)

    return output


# this function projects a complex discrete signal to [0,255] as this interval is used for 8 bit per channel color
# image
def cast_complex_array_to_integers_8bit(signal):
    output = []
    for x in range(len(signal)):
        value = int(signal[x].real)
        if value > 255:
            value = 255
        if value < 0:
            value = 0
        output.append(value)

    return output


# Turns given array into an image and displays it
def array_to_img(img_array):
    image = Image.fromarray(img_array)
    image.show()


# test_fft_removal_on_chapter_4_3_array()
# test_fft_removal_on_pattern()
# test_fft_removal_on_pattern_text()
test_fft_removal_on_pattern_img()

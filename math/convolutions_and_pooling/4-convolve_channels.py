#!/usr/bin/env python3
"""
Performs a strided convolution on images with channels
"""
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a strided convolution on images with channels

    :param images: a numpy.ndarray with shape (m, h, w, c) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    :param kernel: a numpy.ndarray with shape (kh, kw, c) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :param padding: a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    :param stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    :return: numpy.ndarray containing the convolved images
    """
    # Determine the output shape based on the input shape, kernel shape, padding, and stride
    output_height = (images.shape[1] + 2 * padding[0] - kernel.shape[0]) // stride[0] + 1
    output_width = (images.shape[2] + 2 * padding[1] - kernel.shape[1]) // stride[1] + 1

    # Initialize the output array with zeros
    convolved = np.zeros((images.shape[0], output_height, output_width))

    # Loop over the output pixels and compute the convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the input patch centered at the current output pixel
            patch = images[:, i*stride[0]:i*stride[0]+kernel.shape[0],
                          j*stride[1]:j*stride[1]+kernel.shape[1]]

            # Compute the convolution by summing over the spatial dimensions and the channel dimension
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    # Return the convolved images
    return convolved
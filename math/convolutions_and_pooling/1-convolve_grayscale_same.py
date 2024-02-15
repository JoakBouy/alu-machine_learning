#usr/env/bin python3
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    :param images: a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    :param kernel: a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :param padding: a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    :return: a numpy.ndarray containing the convolved images
    """
    # Add padding to the images
    ph, pw = padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))

    # Define the output shape
    output_height = images.shape[1] + 2 * ph
    output_width = images.shape[2] + 2 * pw

    # Initialize the output array
    convolved = np.zeros((images.shape[0], output_height, output_width))

    # Loop over the output pixels and compute the convolution
    for i in range(output_height - kernel.shape[0] + 1):
        for j in range(output_width - kernel.shape[1] + 1):
            # Extract the input patch
            patch = padded_images[:, i:i+kernel.shape[0], j:j+kernel.shape[1]]

            # Compute the convolution
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved
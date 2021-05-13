import math
from matplotlib import image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, io

image1_path = './image1.jpg'
image2_path = './image2.jpg'

def display(img, cmap='default'):
    # Show image
    plt.figure(figsize = (5, 5))
    if (cmap=='H'):
        plt.imshow(img, cmap='hsv')
    elif (cmap=='S'):
        plt.imshow(img, 'Greys')
    elif (cmap=='V'):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### MY CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### MY CODE HERE
    reshaped = image.reshape((image.shape[0] * image.shape[1], 3))
    out = np.array([0.5 * pow(p, 2) for p in reshaped]).reshape((300, 300, 3))
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    print(channel)
    if (channel == 'H'):
        out = hsv[...,0]
    elif (channel == 'S'):
        out = hsv[...,1]
    elif (channel == 'V'):
        out = hsv[...,2]
    print(out)
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    if (channel1=='R'): image1[:, :, 0] = 0
    elif (channel1=='G'): image1[:, :, 1] = 0
    elif (channel1=='B'): image1[:, :, 2] = 0
    
    if (channel2=='R'): image2[:, :, 0] = 0
    elif (channel2=='G'): image2[:, :, 1] = 0
    elif (channel2=='B'): image2[:, :, 2] = 0

    image1[:, 150:, :] = 0
    image2[:, :150, :] = 0
    img1 = image1[:, :, :] + image2[:, :, :]
    out = img1
    # img2 = image2[:, 150:, :]
    # out = img1 + img2
    # print(image1)
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

image1 = load(image1_path)
image2 = load(image2_path)
display(image1)
# display(image2)

# new_image = dim_image(image1)
# display(new_image)

# grey_image = convert_to_grey_scale(image1)
# display(grey_image)

# image_h = hsv_decomposition(image1, 'H')
# image_s = hsv_decomposition(image1, 'S')
# image_v = hsv_decomposition(image1, 'V')

# print("Below is the image with only the H channel.")
# display(image_h, 'H')

# print("Below is the image with only the S channel.")
# display(image_s, 'S')

# print("Below is the image with only the V channel.")
# display(image_v, 'V')

image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)

print(f'{np.sum(image_mixed):.2f}')
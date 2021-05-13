# Setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io


# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# # for auto-reloading external modules
# %load_ext autoreload
# %autoreload 2

# Open image as grayscale
img = io.imread('dog.jpg', as_gray=True)

# # Show image
plt.imshow(img)
plt.axis('off')
plt.title("Isn't he cute?")
plt.show()

#### 1. end
#### A. start

from filters import conv_nested

# Simple convolution kernel.
kernel = np.array(
    [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ]
)

# Create a test image: a white square in the middle
test_img = np.zeros((9, 9))
test_img[3:6, 3:6] = 1

# Run your conv_nested function on the test image
test_output = conv_nested(test_img, kernel)

# Build the expected output
expected_output = np.zeros((9, 9))
expected_output[2:7, 2:7] = 1
expected_output[5:, 5:] = 0
expected_output[4, 2:5] = 2
expected_output[2:5, 4] = 2
expected_output[4, 4] = 3

# Plot the test image
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title('Test image')
plt.axis('off')

# Plot your convolved image
plt.subplot(1, 3, 2)
plt.imshow(test_output)
plt.title('Convolution')
plt.axis('off')

# Plot the expected output
plt.subplot(1, 3, 3)
plt.imshow(expected_output)
plt.title('Expected output')
plt.axis('off')
plt.show()

# Test if the output matches expected output
assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."

#### A. end
#### B. start

#   Simple convolution kernel.
kernel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
)
out = conv_nested(img, kernel)

# Plot the original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

# Plot your convolved image
plt.subplot(2, 2, 3)
plt.imshow(out)
plt.title('Convolution')
plt.axis('off')

# Plot what you should get
solution_img = io.imread('convoluted_dog.jpg', as_gray=True)
plt.subplot(2, 2, 4)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')

plt.show()

# print(f'res : \n{out}')
# print(f'sol : \n{solution_img}')

#### B. end
#### C. start

from filters import zero_pad

pad_width = 20
pad_height = 40

padded_img = zero_pad(img, pad_height, pad_width)

# Plot your padded dog
plt.subplot(1, 2, 1)
plt.imshow(padded_img)
plt.title('Padded dog')
plt.axis('off')

# Plot what you should get
solution_img = io.imread('padded_dog.jpg', as_gray=True)
plt.subplot(1, 2, 2)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')

plt.show()

from filters import conv_fast

t0 = time()
out_fast = conv_fast(img, kernel)
t1 = time()
out_nested = conv_nested(img, kernel)
t2 = time()

# Compare the running time of the two implementations
print(f'conv_nested: took {t2-t1} seconds')
print(f'conf_fast: took {t1-t0} seconds.')

# Plot conv_nested output
plt.subplot(1, 2, 1)
plt.imshow(out_nested)
plt.title('conv_nested')
plt.axis('off')

# Plot conv_fast output
plt.subplot(1, 2, 2)
plt.imshow(out_fast)
plt.title('conv_fast')
plt.axis('off')

plt.show()
# Make sure that the two outputs are the same
if not (np.max(out_fast - out_nested) < 1e-10):
    print('Different outputs! Check your implementation.')

#### C. end
#### 2. start
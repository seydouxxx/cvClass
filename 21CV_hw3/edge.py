"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(kernel))
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i: i+Hk, j: j+Wk] * kernel)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1/(2*np.pi*sigma**2)) * np.exp(-((i - size//2)**2 + (j - size//2)**2) / float(2*sigma**2))

    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0.5,0,-0.5]])

    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0.5],[0],[-0.5]])

    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    gx = partial_x(img)
    gy = partial_y(img)
    G = np.sqrt(gx ** 2 + gy ** 2)
    # G = [[np.sqrt(i**2 + j**2) for i, j in zip(r1, r2)] for r1, r2 in zip(gx, gy)]
    theta = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### YOUR CODE HERE
    for i in range(1, H-1):
        for j in range(1, W-1):
            ang = int(theta[i][j]%360)
            if (ang%180 == 0):
                l = [G[i][j-1], G[i][j+1]]
            elif (ang%180 == 45):
                l = [G[i-1][j-1], G[i+1][j+1]]
            elif (ang%180 == 90):
                l = [G[i-1][j], G[i+1][j]]
            elif (ang%180 == 135):
                l = [G[i-1][j+1], G[i+1][j-1]]
            if G[i,j] >= np.max(l):
                out[i,j] = G[i,j]
            else:
                out[i, j] = 0
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = (img < high) & (img > low)

    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if (i >= 0 and i < H and j >= 0 and j < W):
                if (i != y or j != x):
                    neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    for i in range(1, H-1):
        for j in range(1, W-1):
            neighbor = get_neighbors(j, i, H, W)
            if (weak_edges[i][j] and np.any(edges[x][y] for x, y in neighbor)):
                edges[i][j] = True


    ### END YOUR CODE
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    G, theta = gradient(conv(img, gaussian_kernel(kernel_size, sigma))) 
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE
    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    print(diag_len)
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    # rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):
        for k in range(thetas.shape[0]):
            r = j * cos_t[k] + i * sin_t[k]
            accumulator[int(r + diag_len), k] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas

#   A-1 starts

from edge import conv, gaussian_kernel
from matplotlib import pyplot as plt
from skimage import io

#   Define 3x3 Gaussian kernel with std = 1
kernel = gaussian_kernel(3, 1)
kernel_test = np.array(
    [[0.05854983, 0.09653235, 0.05854983],
    [0.09653235, 0.15915494, 0.09653235],
    [0.05854983, 0.09653235, 0.05854983]]
)
#   Test Gaussian kernel
if not np.allclose(kernel, kernel_test):
    print('Incorrect values! Please check your implementation')

#   A-1 ends
#   A-2 starts

#   Test with different kernel_size and sigma
kernel_size = 5
sigma = 1.4

#   Load image
img = io.imread('iguana.png', as_gray=True)

#   Define 5x5 Gaussian kernel with std = sigma
kernel = gaussian_kernel(kernel_size, sigma)

#   Convolve image with kernel to achieve smoothed effect
smoothed = conv(img, kernel)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed)
plt.title('Smoothed image')
plt.axis('off')

plt.show()

#   A-2 ends
#   B-1 starts

from edge import partial_x, partial_y

#   Test input
I = np.array(
    [[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]]
)

#   Expected outputs
I_x_test = np.array(
    [[0, 0, 0],
    [0.5, 0, -0.5],
    [0, 0, 0]]
)

I_y_test = np.array(
    [[0, 0.5, 0],
    [0, 0, 0],
    [0, -0.5, 0]]
)

#   Compute partial derivatives
I_x = partial_x(I)
I_y = partial_y(I)

#   Test correctness of partial_x and partial_y
if (not np.all(I_x == I_x_test)):
    print('partial_x incorrect')

if (not np.all(I_y == I_y_test)):
    print('partial_y incorrect')

#   Compute parital derivatives of smoothed image
Gx = partial_x(smoothed)
Gy = partial_y(smoothed)

plt.subplot(1, 2, 1)
plt.imshow(Gx)
plt.title('Derivative in x direction')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Gy)
plt.title('Derivative in y direction')
plt.axis('off')

plt.show()

#   B-1 ends
#   B-3 starts

from edge import gradient

G, theta = gradient(smoothed)

if (not np.all(G >= 0)):
    print('Magnitude of gradients should be non-negative')

if (not np.all((theta >= 0) * (theta < 360))):
    print('Direction of gradients should be in range 0 <= theta < 360')

plt.imshow(G)
plt.title('Gradient magnitude')
plt.axis('off')
plt.show()

#   B-3 ends
#   C starts

from edge import non_maximum_suppression

#   Test input
g = np.array(
    [[0.4, 0.5, 0.6],
    [0.3, 0.5, 0.7],
    [0.4, 0.5, 0.6]]
)

#   Print out non-maximum suppressed output
#   Varying theta
for angle in range(0, 180, 45):
    print('Thetas:', angle)
    t = np.ones((3, 3)) * angle
    print(non_maximum_suppression(g, t))

nms = non_maximum_suppression(G, theta)
plt.imshow(nms)
plt.title('Non-maximum suppressed')
plt.axis('off')
plt.show()

plt.subplot(1, 3, 1)
plt.imshow(nms)
plt.axis('off')
plt.title('Your Result')

plt.subplot(1, 3, 2)
reference = np.load('references/iguana_non_max_suppressed.npy')
plt.imshow(reference)
plt.axis('off')
plt.title('Reference')

plt.subplot(1, 3, 3)
plt.imshow(nms - reference)
plt.title('Difference')
plt.axis('off')
plt.show()

#   C ends
#   D starts

from edge import double_thresholding

low_threshold = 0.02
high_threshold = 0.03

strong_edges, weak_edges = double_thresholding(nms, high_threshold, low_threshold)
assert(np.sum(strong_edges & weak_edges) == 0)

edges = strong_edges * 1.0 + weak_edges * 0.5

plt.subplot(1, 2, 1)
plt.imshow(strong_edges)
plt.title('Strong Edges')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges)
plt.title('Strong+Weak Edges')
plt.axis('off')

plt.show()

#   D ends
#   E starts

from edge import get_neighbors, link_edges

test_strong = np.array(
    [[1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]],
    dtype=np.bool
)

test_weak = np.array(
    [[0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0]],
    dtype=np.bool
)

test_linked = link_edges(test_strong, test_weak)

plt.subplot(1, 3, 1)
plt.imshow(test_strong)
plt.title('Strong edges')

plt.subplot(1, 3, 2)
plt.imshow(test_weak)
plt.title('Weak edges')

plt.subplot(1, 3, 3)
plt.imshow(test_linked)
plt.title('Linked edges')
plt.show()

edges = link_edges(strong_edges, weak_edges)

plt.imshow(edges)
plt.axis('off')
plt.show()

plt.subplot(1, 3, 1)
plt.imshow(edges)
plt.axis('off')
plt.title('Your result')

plt.subplot(1, 3, 2)
reference = np.load('references/iguana_edge_tracking.npy')
plt.imshow(reference)
plt.axis('off')
plt.title('Reference')

plt.subplot(1, 3, 3)
plt.imshow(edges ^ reference)
plt.title('Difference')
plt.axis('off')
plt.show()

#   E ends
#   F starts

from edge import canny

#   Load image
img = io.imread('iguana.png', as_gray=True)

#   Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
print(edges.shape)

plt.subplot(1, 3, 1)
plt.imshow(edges)
plt.axis('off')
plt.title('Your result')

plt.subplot(1, 3, 2)
reference = np.load('references/iguana_canny.npy')
plt.imshow(reference)
plt.axis('off')
plt.title('Reference')

plt.subplot(1, 3, 3)
plt.imshow(edges ^ reference)
plt.title('Difference')
plt.axis('off')
plt.show()

#   F ends
#   2.A starts

from edge import canny

#   Load image
img = io.imread('road.jpg', as_gray=True)

#   Run canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)

plt.subplot(211)
plt.imshow(img)
plt.axis('off')
plt.title('Input image')

plt.subplot(212)
plt.imshow(edges)
plt.axis('off')
plt.title('Edges')
plt.show()

#   2.A ends
#   2.B starts

H, W = img.shape

#   Generate mask for ROI(Region of Interest)
mask = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if (i>(H/W) * j and i > -(H/W) * j + H):
            mask[i, j] = i

#   Extract edges in ROI
roi = edges * mask

plt.subplot(1, 2, 1)
plt.imshow(mask)
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(roi)
plt.title('Edges in ROI')
plt.axis('off')
plt.show()

#   2.B ends
#   2.C starts

from edge import hough_transform

#   Perform Hough transform on the ROI
acc, rhos, thetas = hough_transform(roi)

#   Coordinates for right lane
xs_right = []
ys_right = []

#   Coordinates for left lane
xs_left = []
ys_left = []

for i in range(20):
    idx = np.argmax(acc)
    r_idx = idx // acc.shape[1]
    t_idx = idx % acc.shape[1]
    acc[r_idx, t_idx] = 0   #   zero out the max value in accumulator

    rho = rhos[r_idx]
    theta = thetas[t_idx]

    #   Transform a point in Hough space to a line in xy-space.
    a = -(np.cos(theta) / np.sin(theta))    #   slope of the line
    b = (rho/np.sin(theta)) #   y-intersect of the line

    #   Break if both right and left lanes are detected
    if (xs_right and xs_left):
        break

    if (a<0):   #   Left lane
        if (xs_left):
            continue
        xs = xs_left
        ys = ys_left
    
    else:   #   Right lane
        if (xs_right):
            continue
        xs = xs_right
        ys = ys_right
    
    for x in range(img.shape[1]):
        y = a * x + b
        if (y > img.shape[0] * 0.6 and y < img.shape[0]):
            xs.append(x)
            ys.append(int(round(y)))

plt.imshow(img)
plt.plot(xs_left, ys_left, linewidth=5.0)
plt.plot(xs_right, ys_right, linewidth=5.0)
plt.axis('off')
plt.show()
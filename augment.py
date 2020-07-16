import os, sys
from glob import glob
import numpy as np
from skimage import feature
from skimage import filters
from matplotlib.pyplot import imread, imsave, imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import draw


def outer_contour(img):
    img = rgb2gray(img)
    img = feature.canny(img)

    # find left- and right-most contour pixels
    left = np.zeros(img.shape[0], dtype=int)
    right = np.zeros(img.shape[0], dtype=int)
    for y in range(img.shape[0]):
        contour_coords = np.where(img[y, :])[0]
        if len(contour_coords) > 0:
            right[y] = np.max(contour_coords)
            left[y] = np.min(contour_coords)

    cont_img = np.zeros_like(img)

    # draw lines between consecutive left- and right-most pixels (separately)
    for y in range(1, img.shape[0]):
        if right[y - 1] > 0 and right[y] > 0:
            line = draw.line(y - 1, right[y - 1], y, right[y])
            cont_img[line[0], line[1]] = 1.
        if left[y - 1] > 0 and left[y] > 0:
            line = draw.line(y - 1, left[y - 1], y, left[y])
            cont_img[line[0], line[1]] = 1.

    # draw line between top and bottom left-most and right-most pixels
    top = np.min(np.where(right > 0)[0])
    bottom = np.max(np.where(right > 0)[0])
    line = draw.line(top, left[top], top, right[top])
    cont_img[line[0], line[1]] = 1.
    line = draw.line(bottom, left[bottom], bottom, right[bottom])
    cont_img[line[0], line[1]] = 1.

    return cont_img


img = imread('data/dl/startrek/white-bg-only/5.jpg')
outer_contour(img)

import os, sys, shutil
from glob import glob
import numpy as np
from skimage import feature
from skimage import filters
from matplotlib.pyplot import imread, imsave, imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import draw
from skimage.transform import rescale, resize
from skimage.morphology import dilation


def oob(a, x, y):  # out of bounds
    return y >= a.shape[0] or x >= a.shape[1] or y < 0 or x < 0


def oob_v(a, x, y):  # out of bounds vectorized
    return (y >= a.shape[0]) | (x >= a.shape[1]) | (y < 0) | (x < 0)


def discover(img, threshold, to_discover, covered):
    grp = set()

    while len(to_discover) > 0:
        x, y = to_discover.pop()
        if oob(img, x, y) or (x, y) in covered:
            continue

        if img[y, x] >= threshold:
            grp.add((x, y))
            to_discover.add((x + 1, y))
            to_discover.add((x - 1, y))
            to_discover.add((x, y + 1))
            to_discover.add((x, y - 1))

        covered.add((x, y))

    return grp


def overlay_groups_on_img(groups, base_img, col=None):
    col = np.random.random((len(groups) + 1, 3)) if col is None else col
    col_img = rgb2gray(base_img) if len(base_img.shape) == 2 else base_img.copy()
    for i, grp in enumerate(groups):
        if len(grp) == 0:
            continue
        coords = np.array(list(grp))
        col_img[coords[:, 1], coords[:, 0]] = col[i] if len(col.shape) > 1 else col  # swapped x & y

    return col_img


def outer_contour(img, dilate=0):

    # take img contour
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

    # dilate contour
    for _ in range(dilate):
        cont_img = dilation(cont_img)

    return cont_img, left, right, top, bottom


def color_bg(img, col):
    img = img.copy()
    corners = {(0, 0), (img.shape[0] - 1, img.shape[1] - 1), (0, img.shape[1] - 1), (img.shape[0] - 1, 0)}
    bg = discover(rgb2gray(img), 0.95, to_discover=corners, covered=set())
    colored = overlay_groups_on_img([bg], img, col=col)
    return colored


def fit_to_box(img, h, w, fill_col):  # img has to be uint8
    hratio = h / img.shape[0]
    wratio = w / img.shape[1]
    scale = min(hratio, wratio)

    img = rescale(img, scale, anti_aliasing=True, multichannel=True if len(img.shape) == 3 else False)

    shape = (h, w) if len(img.shape) == 2 else (h, w, img.shape[2])
    boxed_img = np.ones(shape) * fill_col

    boxed_img[(h - img.shape[0]) // 2:(h - img.shape[0]) // 2 + img.shape[0],
              (w - img.shape[1]) // 2:(w - img.shape[1]) // 2 + img.shape[1]] = img

    return boxed_img


def horizontal_crop(img, n, left_frame_size=0):
    stride = img.shape[1] // n
    left_frame = np.ones((img.shape[0], left_frame_size, 3), dtype=img.dtype) * img[0, 0]
    return [np.concatenate([left_frame, img[:, left:]], axis=1) for left in range(0, img.shape[1] - stride, stride)]


def prep_img(img, h, w, dilate, bgcol=np.array([0, 0, 0])):  # for pytorch-CycleGAN-and-pix2pix
    assert img.dtype == np.uint8

    img = color_bg(img, bgcol)
    cont_img, _, _, _, _ = outer_contour(img, dilate)

    img = fit_to_box(img, h, w, bgcol)
    cont_img = fit_to_box((cont_img * 255).astype(np.uint8), h, w, 0)
    cont_img = np.tile(np.expand_dims(cont_img, -1), (1, 1, 3))

    return np.concatenate([img, cont_img], axis=1)


if __name__ == '__main__':

    # img = imread('data/dl/startrek/white-bg-only/5.jpg')
    # ready = prep_img(img, 256, 256, 3)
    # imshow(ready)
    # plt.show()

    src_wc = 'data/raw/spaceship/*.jpg'
    out_dir = 'data/augm'
    h, w = 256, 256
    dilate = 2  # number of times the contour is dilated
    bgcol = np.array([0, 0, 0])
    train_ratio, val_ratio, test_ratio = .75, .2, .05
    ncrop = 4  # number of horizontal crops
    left_frame_after_crop = 10  # size in pixels

    imgpaths = np.random.permutation([f for f in glob(src_wc)])
    ntrain, nval = int(test_ratio * len(imgpaths)), int(val_ratio * len(imgpaths))
    ntest = len(imgpaths) - ntrain - nval

    # clear out_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    for d in ['train', 'val', 'test']:
        os.makedirs(f'{out_dir}/{d}')

    for i, impath in enumerate(imgpaths):

        subf = 'train'
        if ntrain < i < ntrain + nval:
            subf = 'val'
        elif ntrain + nval < i:
            subf = 'test'

        img = imread(impath)

        # augmentations
        augms = [img]
        if subf == 'train':
            augms = horizontal_crop(img, ncrop, left_frame_after_crop)
            augms.extend([np.fliplr(a) for a in augms])
            augms.extend([np.flipud(a) for a in augms])

        # save augmented images
        for a, augm_img in enumerate(augms):
            augm_img = prep_img(augm_img, h, w, dilate, bgcol)
            imsave(f'{out_dir}/{subf}/{i}_{a}.jpg', augm_img)

        if i % 100 == 0:
            print(f'{i}/{len(imgpaths)}')

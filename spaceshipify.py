import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage import draw
from skimage.morphology import dilation
from skimage.transform import resize
from augment import fit_to_box, fill_inside_contour
from matplotlib.pyplot import imread, imsave, imshow


def code_outline(code, tolerate_0s=False):
    lines = code.split('\n')

    left, right = [], []
    for l, line in enumerate(lines):
        spaces = 0
        for c in line:
            if c == ' ':
                spaces += 1
            else:
                break
        left.append(spaces)

        r = len(line)
        if not tolerate_0s and r == 0:
            # copy the len of the closest and shortest nonzero len line
            for d in range(len(lines)):
                r1 = len(lines[l - d]) if l - d >= 0 and len(lines[l - d]) > 0 else 0
                r2 = len(lines[l + d]) if l + d < len(lines) and len(lines[l + d]) > 0 else 0
                if max(r1, r2) > 0:
                    r = min(r1, r2) if min(r1, r2) > 0 else max(r1, r2)
                    break

        right.append(r)

    return left, right


def dilate_array(a, n):
    return np.tile(np.expand_dims(a, -1), (1, n)).ravel()


def smooth_contour(cont, window_len, order=2, mode='nearest'):
    window_len = window_len if window_len % 2 == 1 else window_len + 1
    return np.clip(signal.savgol_filter(cont, window_len, order, mode=mode), 0, np.inf)


# create whole contour from left and right
def draw_contour(left, right, char_w, dilate, pad, fill=True):  # left and right have to be int pixel coordinates
    assert len(left) == len(right)

    left *= char_w
    right *= char_w

    cont_img = np.zeros((len(left), int(np.max(right)) + 1))
    for y in range(1, len(left)):
        if right[y - 1] > 0 and right[y] > 0:
            line = draw.line(y - 1, right[y - 1], y, right[y])
            cont_img[line[0], line[1]] = 1.
        line = draw.line(y - 1, left[y - 1], y, left[y])
        cont_img[line[0], line[1]] = 1.

    # draw line between top and bottom left-most and right-most pixels
    top = np.min(np.where(right > 0)[0])
    bottom = np.max(np.where(right > 0)[0])
    line = draw.line(top, left[top], top, right[top])
    cont_img[line[0], line[1]] = 1.
    line = draw.line(bottom, left[bottom], bottom, right[bottom])
    cont_img[line[0], line[1]] = 1.

    # add cushion
    cont_img = np.pad(cont_img, [(pad, pad), (pad, pad)])  # x,y on both directions
    cont_img = cont_img.astype(np.uint8) * 255
    left += pad
    right += pad
    left, right = np.pad(left, (pad, pad)), np.pad(right, (pad, pad))

    # dilate contour
    for _ in range(dilate):
        cont_img = dilation(cont_img)

    if fill:
        cont_img = fill_inside_contour(cont_img, left, right, dilate, all_components=True)

    return np.tile(np.expand_dims(cont_img, -1), (1, 1, 3))


def code2contour(code, box_size=256, order=3, dilate=1, pad=5, fill_cont=True):

    left_cont, right_cont = code_outline(code)
    line_h = box_size // len(right_cont)
    char_w = line_h // 3
    window_len = int(line_h * 3)

    left_cont = dilate_array(left_cont, line_h)
    right_cont = dilate_array(right_cont, line_h)

    left_cont = smooth_contour(left_cont, window_len // 2, order)
    right_cont = smooth_contour(right_cont, window_len // 2, order)

    # pull closer to zero if the left is close to 0 on avg or if it has a high variance
    push_left_to_zero = np.mean(left_cont) / np.max(left_cont) / np.std(left_cont) * 5
    left_cont *= push_left_to_zero
    # left cannot be higher than right, it should strictly be smaller by a little at least
    left_cont = np.clip(np.stack([left_cont, right_cont - box_size / 15], axis=-1).min(axis=-1), 0, np.inf)

    contour = draw_contour(left_cont.astype(int), right_cont.astype(int), char_w, dilate, pad, fill_cont)
    # orig_shape = contour.shape  # keep for later so the generated ship can be deformed back, if needed at all

    # contour = (resize(contour, (box_size, box_size)) * 255).astype(np.uint8)
    contour = fit_to_box(contour, box_size, box_size, np.array([0, 0, 0]))

    return contour


CODE1 = """
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
"""

CODE2 = """
def code_contour(code, tolerate_0s=False):
    lines = code.split('\n')

    left, right = [], []
    for l, line in enumerate(lines):
        spaces = 0
        for c in line:
            if c == ' ':
                spaces += 1
            else:
                break
        left.append(spaces)

        r = len(line)
        if not tolerate_0s and r == 0:
            # copy the len of the closest nonzero len line
            for d in range(len(lines)):
                if l - d >= 0 and len(lines[l - d]) > 0:
                    r = len(lines[l - d])
                    break
                elif l + d < len(lines) and len(lines[l + d]) > 0:
                    r = len(lines[l + d])
                    break
        right.append(r)

    return left, right
"""

CODE3 = """
box_size = 256
left_cont, right_cont = code_contour(code1)

line_h = box_size // len(right_cont)
char_w = line_h // 3
window_len = int(line_h * 3)
order = 1
dilate = 1
pad = 10
push_left_to_zero = np.mean(left_cont)  # .5 means half-way between 0 and the actual value, 0. is full zero

left_cont = dilate_array(left_cont, line_h)
right_cont = dilate_array(right_cont, line_h)
print(left_cont)
left_cont = smooth_contour(left_cont, window_len//2, order)
right_cont = smooth_contour(right_cont, window_len//2, order)
# left_cont = egyenget_contour(left_cont, window_len//2)
# right_cont = egyenget_contour(right_cont, window_len//2)

left_cont *= push_left_to_zero

contour = draw_contour(left_cont.astype(int), right_cont.astype(int), char_w, dilate, pad)
orig_shape = contour.shape  # keep for later so the generated ship can be deformed back, if needed at all

# contour = (resize(contour, (box_size, box_size)) * 255).astype(np.uint8)
contour = fit_to_box(contour, box_size, box_size, np.array([0, 0, 0]))

plt.imshow(contour)
plt.show()

print(len(right_cont))
plt.plot(right_cont)
"""

CODE4 = """
def code2contour(code, box_size=256, order=3, dilate=1, pad=5):

    left_cont, right_cont = code_outline(code)
    line_h = box_size // len(right_cont)
    char_w = line_h // 3
    window_len = int(line_h * 3)
    
    left_cont = dilate_array(left_cont, line_h)
    right_cont = dilate_array(right_cont, line_h)

    left_cont = smooth_contour(left_cont, window_len // 2, order)
    right_cont = smooth_contour(right_cont, window_len // 2, order)

    # pull closer to zero if the left is close to 0 on avg or if it has a high variance
    push_left_to_zero = np.mean(left_cont) / np.max(left_cont) / np.std(left_cont) * 6
    left_cont *= push_left_to_zero

    contour = draw_contour(left_cont.astype(int), right_cont.astype(int), char_w, dilate, pad)
    # orig_shape = contour.shape  # keep for later so the generated ship can be deformed back, if needed at all

    # contour = (resize(contour, (box_size, box_size)) * 255).astype(np.uint8)
    contour = fit_to_box(contour, box_size, box_size, np.array([0, 0, 0]))
    
    return contour
"""


CODE5 = """
if __name__ == '__main__':

    out_dir = 'data/augm/test'
    box_size = 256
    order = 1
    dilate = 1
    pad = 10
    fill_cont = True

    for c, code in enumerate([CODE1, CODE2, CODE3, CODE4]):
        contour = code2contour(code, box_size, order, dilate, pad, fill_cont)
        # plt.imshow(contour)
        # plt.show()

        prep_contour = np.concatenate([np.zeros_like(contour), contour], axis=1)
        imsave(f'{out_dir}/code_{c}.jpg', prep_contour)

    # python3.6 test.py <SAME PARAMS AS FOR train.py>
"""


if __name__ == '__main__':

    out_dir = 'data/augm/test'
    box_size = 256
    order = 1
    dilate = 1
    pad = 10
    fill_cont = True

    for c, code in enumerate([CODE1, CODE2, CODE3, CODE4, CODE5]):
        contour = code2contour(code, box_size, order, dilate, pad, fill_cont)
        # plt.imshow(contour)
        # plt.show()

        prep_contour = np.concatenate([np.zeros_like(contour), contour], axis=1)
        imsave(f'{out_dir}/code_{c}.jpg', prep_contour)

    # python3.6 test.py <SAME PARAMS AS FOR train.py>

import sys
import io
import argparse
import re
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage import draw
from skimage.morphology import dilation
from skimage.transform import resize
from augment import fit_to_box, fill_inside_contour
from matplotlib.pyplot import imread, imsave, imshow
from stable_diff import StableSpaceship
import ast


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


def code2contour(code, box_size=256, order=3, dilate=1, pad=5, fill_cont=True, set_left=None,
                 do_fit_to_box=True, ret_pix_sizes=False):

    left_cont, right_cont = code_outline(code)
    line_h = box_size // len(right_cont)
    char_w = line_h // 3
    window_len = max(3, int(line_h * .5))

    left_cont = dilate_array(left_cont, line_h)
    right_cont = dilate_array(right_cont, line_h)

    left_cont = smooth_contour(left_cont, window_len, order)
    right_cont = smooth_contour(right_cont, window_len, order)

    if set_left is not None:
        left_cont = np.full_like(left_cont, set_left)
    else:
        # pull closer to zero if the left is close to 0 on avg or if it has a high variance
        push_left_to_zero = np.mean(left_cont) / np.max(left_cont) / np.std(left_cont) * 5
        left_cont *= push_left_to_zero

    # left cannot be higher than right, it should strictly be smaller by a little at least
    left_cont = np.clip(np.stack([left_cont, right_cont - box_size / 15], axis=-1).min(axis=-1), 0, np.inf)

    contour = draw_contour(left_cont.astype(int), right_cont.astype(int), char_w, dilate, pad, fill_cont)
    # orig_shape = contour.shape  # keep for later so the generated ship can be deformed back, if needed at all

    # contour = (resize(contour, (box_size, box_size)) * 255).astype(np.uint8)
    if do_fit_to_box:
        contour = fit_to_box(contour, box_size, box_size, np.array([0, 0, 0]))

    if ret_pix_sizes:
        return contour, (line_h, char_w)
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
    lines = code.split('\\n')

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


def code_contour(code, tolerate_0s=False):
    lines = code.split('\\n')

    left, right = [], []
    
    
    def code_contour(code, tolerate_0s=False):
        lines = code.split('\\n')
    
        left, right = [], []
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


CODE6 = """
#!/bin/bash

name="c2s_144"
dataroot="data/augm"
dataroot_p2p="../${dataroot}"  # pix2pix is in a subfolder
load_size=256
n_epochs=150  # keep lr at init for this long
n_epochs_decay=150  # number of epochs to 0 lr
batch_size=32
ngf=144
ndf=128
n_layers_D=4

# generate data
python3.6 augment.py ${dataroot}
python3.6 spaceshipify.py ${dataroot}

# run training and testing
cd pytorch-CycleGAN-and-pix2pix
python3.6 train.py --dataroot ${dataroot_p2p} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size}
    --n_epochs ${n_epochs} --n_epochs_decay ${n_epochs_decay} --model pix2pix --direction BtoA --batch_size ${batch_size}
    --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf} --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D}
    --display_winsize ${load_size} --save_epoch_freq 50 --display_freq 1000
python3.6 test.py --dataroot ${dataroot_p2p} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size}
    --model pix2pix --direction BtoA --batch_size ${batch_size} --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf}
    --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D} --display_winsize ${load_size} --num_test 100000
"""


# TODO break the code further into blocks according to double line breaks - have it as option here


def example_run():
    box_size = 512
    order = 1
    dilate = 1
    pad = 10
    fill_cont = True
    set_left = 1  # or None  # TODO

    sss = StableSpaceship()
    steps = 250
    prompt = 'sideview of a spaceship from star wars with silver and blue colors. ' \
             'high definition aesthetic 3d model'

    for c, code in enumerate([CODE1, CODE2, CODE3, CODE4, CODE5, CODE6]):
        contour = code2contour(code, box_size, order, dilate, pad, fill_cont, set_left)

        ship = sss.run(contour, prompt=prompt, steps=steps)

        plt.imshow(np.concatenate([resize(contour, ship.shape), ship], axis=1))
        plt.show()

        plt.imsave('contour_example.png', contour)
        plt.imsave('spaceship_example.png', ship)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spaceshipifies Python scripts')
    parser.add_argument('code_path', help='path to Python script', default='tmp.py')
    parser.add_argument('out_path', help='destination of generated spaceship image', default='out/armada.png')
    # parser.add_argument('--line', help='editor line height in pixels', type=int)
    parser.add_argument('-st', '--steps', help='number of refinement steps to take', type=int, default=200)
    parser.add_argument('-si', '--size', help='size of the output image', type=int, default=512)
    parser.add_argument('-p', '--prompt', help='style description of spaceship', default=
                        'high definition aesthetic sideview of the front half of a single spaceship from star wars with blue colors')

    args = parser.parse_args()

    # contour generation settings
    box_size = args.size
    order = 1
    dilate = 1
    pad = 0
    fill_cont = True
    set_left = 1  # or None
    # line_height = args.line

    # stable diffusion settings
    prompt = args.prompt
    steps = args.steps

    # load code and find function
    with open(args.code_path, 'rt') as f:
        code_txt = f.read()

    # # find functions w/ regex
    # reg = re.compile(r'def (\w+)\s*\((.*?)\):')
    # for m in reg.finditer(code_txt):
    #     print(m.start(), m.group())

    hide_stdout = sys.stdout
    hide_stderr = sys.stderr
    # sys.stdout = io.BytesIO()  # TODO suppress stdout
    # sys.stderr = io.BytesIO()  # TODO suppress stderr
    try:
        # find functions w/ ast
        ast_parsed = ast.parse(code_txt)
        funs = [n for n in ast.walk(ast_parsed) if type(n) == ast.FunctionDef]
        fun_spans = [(n.lineno, n.end_lineno) for n in funs]
        src_segments = [ast.get_source_segment(code_txt, n, padded=True) for n in funs]

        # select non-overlapping functions (nested functions don't get their own spaceship)
        invalid_spans_i = set()
        for fi in range(0, len(fun_spans) - 1):
            for fj in range(fi + 1, len(fun_spans)):
                si, sj = fun_spans[fi], fun_spans[fj]
                if si[0] <= sj[0] and sj[1] <= si[1]:
                    invalid_spans_i.add(fj)

        fun_spans = [sp for i, sp in enumerate(fun_spans) if i not in invalid_spans_i]
        src_segments = [ss for i, ss in enumerate(src_segments) if i not in invalid_spans_i]

        if len(fun_spans) == 0:
            print('{"liftoff": "nah", "whynah": "nofun"}', file=hide_stdout)
            exit(0)

        # load stable diffusion model
        sss = StableSpaceship()

        # generate spaceships for each function
        contours, ships = [], []
        pix_sizes = []  # line height and character widths calculated individually for each spaceship

        for span, segment in zip(fun_spans, src_segments):
            contour, (line_h, char_w) = code2contour(segment, box_size, order, dilate, pad, fill_cont, set_left,
                                                     do_fit_to_box=False, ret_pix_sizes=True)
            orig_shape = contour.shape
            ship = sss.run(resize(contour, (box_size, box_size)), prompt=prompt, steps=steps)
            ship = resize(ship, orig_shape)

            ships.append(ship)
            contours.append(contour)
            pix_sizes.append([line_h, char_w])

        # resize spaceships to have the same line height and char width
        pix_sizes = np.array(pix_sizes, dtype=float)
        shared_line_h = int(pix_sizes[:, 0].mean())
        shared_char_w = int(pix_sizes[:, 1].mean())
        resize_h_to = [int((span[1] - span[0] + 1) * shared_line_h) for span in fun_spans]
        resize_w_by = shared_char_w / pix_sizes[:, 1]
        ships = [resize(ship, [resize_h_to[i], int(ship.shape[1] * resize_w_by[i])])
                 for i, ship in enumerate(ships)]

        # combine spaceships
        armada_h = (code_txt.count('\n') + 1 + 1) * shared_line_h  # +1 for safety
        armada_w = max([ship.shape[1] for ship in ships])
        armada = np.zeros((armada_h, armada_w, ships[0].shape[-1]), dtype=ships[0].dtype)
        for ship, span in zip(ships, fun_spans):
            h_start = (span[0] - 1) * shared_line_h  # 1 -> 0-indexing
            h_end = span[1] * shared_line_h  # -1 as above and +1 for including the last line
            armada[h_start:h_end, 0:ship.shape[1], :] = ship

        plt.imsave(args.out_path, armada)
        print('{"liftoff": "success", "line_h": %d, "char_w": %d}' % (shared_line_h, shared_char_w), file=hide_stdout)

    except SyntaxError:  # this ain't a debugger
        print('{"liftoff": "nah", "whynah": "syntaxerror"}', file=hide_stdout)
        exit(1)

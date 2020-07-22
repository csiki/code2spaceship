import torch
import numpy as np
import matplotlib.pyplot as plt

# TODO try higher img dim (no change in options necessary)

# if --netD is set to "n_layers" and --n_layers_D to different numbers, the patchgan patch size changes:
#   1 layer: 16 patch size; 2: 34; 3: 70 (default); 4: 142; 5: 286
#   matlab script to calculate this: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m

# train:
#   code2spaceships: python3.6 train.py --dataroot ../data/augm --gpu_ids 0,1 --name code2spaceships --model pix2pix --direction BtoA --batch_size 32 --norm batch --netG unet_256 --ngf 128 --ndf 128 --preprocess none --no_flip
#   code2spaceships_142patch: python3.6 train.py --dataroot ../data/augm --gpu_ids 0,1 --name code2spaceships_142patch --model pix2pix --direction BtoA --batch_size 32 --norm batch --netG unet_256 --ngf 128 --ndf 128 --preprocess none --no_flip --netD n_layers --n_layers_D 4
#   code2spaceships_142patch_bw: same as above TODO change the number of channels to 1 only (grayscale)
# view results: python3.6 -m visdom.server
#   then open http://localhost:8097

# taken out: --load_size 256 --crop_size 256

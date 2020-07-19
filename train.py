import torch
import numpy as np
import matplotlib.pyplot as plt

# TODO try --netD w/ pixel OR n_layers; for the latter then define --n_layers_D to be >3
# TODO try --ngf to be >64, same for --ndf
# TODO try higher img dim (no change in options necessary)

# train: python3.6 train.py --dataroot ../data/augm --name code2spaceships --model pix2pix --direction BtoA --batch_size 16 --norm batch --netG unet_256 --preprocess none --no_flip
# view results: python3.6 -m visdom.server
#   then open http://localhost:8097

# taken out: --load_size 256 --crop_size 256

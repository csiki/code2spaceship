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
python3.6 train.py --dataroot ${dataroot_p2p} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size} \
    --n_epochs ${n_epochs} --n_epochs_decay ${n_epochs_decay} --model pix2pix --direction BtoA --batch_size ${batch_size} \
    --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf} --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D} \
    --display_winsize ${load_size} --save_epoch_freq 50 --display_freq 1000
python3.6 test.py --dataroot ${dataroot_p2p} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size} \
    --model pix2pix --direction BtoA --batch_size ${batch_size} --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf} \
    --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D} --display_winsize ${load_size} --num_test 100000

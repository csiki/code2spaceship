#!/bin/bash

name="c2s_512"
dataroot="../data/augm"
load_size=512
n_epochs=200  # keep lr at init for this long
n_epochs_decay=200  # number of epochs to 0 lr
batch_size=8
ngf=128
ndf=128
n_layers_D=5

python3.6 augment.py
cd pytorch-CycleGAN-and-pix2pix
python3.6 train.py --dataroot ${dataroot} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size} --n_epochs ${n_epochs} --n_epochs_decay ${n_epochs_decay} --model pix2pix --direction BtoA --batch_size ${batch_size} --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf} --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D} --display_winsize ${load_size}
python3.6 test.py --dataroot ${dataroot} --gpu_ids 0,1 --name ${name} --load_size ${load_size} --crop_size ${load_size} --n_epochs ${n_epochs} --n_epochs_decay ${n_epochs_decay} --model pix2pix --direction BtoA --batch_size ${batch_size} --norm batch --netG unet_256 --ngf ${ngf} --ndf ${ndf} --preprocess none --no_flip --netD n_layers --n_layers_D ${n_layers_D} --display_winsize ${load_size} --num_test 100000

#!/bin/bash

source ${HOME}/nas01home/linux/anaconda3/etc/profile.d/conda.sh
conda activate sifi
export PYTHONPATH=/nas01/homes/ogita23-1000068/sifigan/SiFiGAN_VAE/sifigan
export HYDRA_FULL_ERROR=1
sifigan-vae-train generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp/vaevocoder_f0_shifter_32
#!/bin/bash

source ${HOME}/nas01home/linux/anaconda3/etc/profile.d/conda.sh
conda activate sifi
export PYTHONPATH=/nas01/homes/ogita23-1000068/sifigan/SiFiGAN_VAE/sifigan
export HYDRA_FULL_ERROR=1
##python3 eval.py ++method_dir=exp/vaevocoder_contentvec_aug_2_cossim1.0/wav/500000
#python3 eval.py ++method_dir=exp/vaevocoder_contentvec_aug_2_2.0/wav/500000
##python3 eval.py ++method_dir=exp/vaevocoder_contentvec_aug_2_5.0/wav/500000#
#python3 eval.py ++method_dir=exp/vaevocoder_contentvec_aug_2_10.0/wav/500000
#python3 eval.py ++method_dir=exp/vaevocoder_contentvec_aug_2_20.0/wav/500000
python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_20_0.0002/wav/100000 ++exp_name=exp_grid_mse
#python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_25_0.0002/wav/100000 ++exp_name=exp_grid_mse
#python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_30_0.0002/wav/100000 ++exp_name=exp_grid_mse
#python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_35_0.0002/wav/100000 ++exp_name=exp_grid_mse
#python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_40_0.0002/wav/100000 ++exp_name=exp_grid_mse
#python3 eval.py ++method_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_45_0.0002/wav/100000 ++exp_name=exp_grid_mse
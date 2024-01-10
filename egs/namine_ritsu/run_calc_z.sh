#!/bin/bash

export HYDRA_FULL_ERROR=1
#python3 calc_latent_mse.py ++out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_20_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_20_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_25_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_30_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_35_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_40_0.0002 ++checkpoint_steps=100000
python3 calc_latent_mse.py ++out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_45_0.0002 ++checkpoint_steps=100000
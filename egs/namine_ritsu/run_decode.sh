#!/bin/bash

export HYDRA_FULL_ERROR=1
#sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=500000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp/vaevocoder_base
#sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=50000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_35_0.0002
#sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=500000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp/vaevocoder_contentvec_aug_2_2.0
#sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=500000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp/vaevocoder_contentvec_aug_2_5.0
#sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=500000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp/vaevocoder_contentvec_aug_2_10.0

sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=100000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_35_0.0002
sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=100000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_40_0.0002
sifigan-vae-decode generator=vaeganvocoder data=namine_ritsu checkpoint_steps=100000 f0_factors=[0.5,1.0,1.25,1.5,1.75,2.0] out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg_45_0.0002
#!/bin/bash

#source ${HOME}/nas01home/linux/anaconda3/etc/profile.d/conda.sh
#conda activate sifi
#export PYTHONPATH=/nas01/homes/ogita23-1000068/sifigan/SiFiGAN_VAE/sifigan
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
#sifigan-vae-train generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp/vaevocoder_contentvec_aug_2_30.0 ++train.lambda_latent=30.0 \
#++train.train_max_steps=1000000 ++train.save_interval_steps=50000 ++train.eval_interval_steps=10000 ++train.log_interval_steps=2000 ++data.batch_size=8
#sifigan-vae-train generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp/vaevocoder_shifter_aug_3 \
#++train.train_max_steps=1000000 ++train.save_interval_steps=50000 ++train.eval_interval_steps=10000 ++train.log_interval_steps=2000 ++data.batch_size=32

# for grid-search
# sifigan-vae-train -m generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim ++generator.decoder_params.channels=216 ++train.latent_loss.loss_type=cossim ++train.train_max_steps=100000 ++data.batch_size=3 '++train.lambda_latent=30,40,50' '++train.generator_optimizer.lr=2.0e-6,2.0e-5,2.0e-4,2.0e-3,2.0e-2,2.0e-1'

# for grid-search
#sifigan-vae-train-avg generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg ++generator._target_=sifigan.models.VAEGANVocoder_avg ++train.latent_loss.loss_type=mse ++train.train_max_steps=100000 ++data.batch_size=16 '++train.lambda_latent=35'
#sifigan-vae-train-avg generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp_grid_mse/vaevocoder_contentvec_aug_2_mse_avg ++generator._target_=sifigan.models.VAEGANVocoder_avg ++train.latent_loss.loss_type=mse ++train.train_max_steps=100000 ++data.batch_size=16 '++train.lambda_latent=40'
sifigan-vae-train-avg generator=vaeganvocoder discriminator=univnet train=vaeganvocoder data=namine_ritsu out_dir=exp_grid_cossim/vaevocoder_contentvec_aug_2_cossim_avg_2.4 ++generator._target_=sifigan.models.VAEGANVocoder_avg ++train.latent_loss.loss_type=cossim ++train.train_max_steps=500000 ++data.batch_size=16 '++train.lambda_latent=20'
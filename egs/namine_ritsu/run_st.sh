#!/bin/bash

HYDRA_FULL_ERROR=1
#sifigan-vae-extract-features audio=data/scp/namine_ritsu_dev_aug_2_0.4_2.5.scp
#sifigan-vae-extract-features-base audio=data/scp/namine_ritsu_eval.scp
sifigan-vae-compute-statistics feats=data/scp/namine_ritsu_eval_f0.list stats=data/stats/namine_ritsu_eval_f0.joblib
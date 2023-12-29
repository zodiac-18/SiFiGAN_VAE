# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate mse of latent space.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""

import os
from logging import getLogger
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
import librosa
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from sifigan.datasets import FeatDataset

# A logger for this file
logger = getLogger(__name__)

class ObjectiveScore:
    def __init__(
        self,
        sr=24000,  # Sampling rate
        n_fft=2048,  # FFT size
        frame_period=10.0,  # Hop size in [ms]
        f0_floor=100.0,  # Minimum F0 for F0 estimation [Hz]
        f0_ceil=1000.0,  # Maximum F0 for F0 estimation [Hz]
        n_mels=40,  # Number of bins for mel-cepstram extraction [Hz]
        exclude_0th_component_in_LSD=True,
        exclude_0th_component_in_MCD=True,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.frame_period = frame_period
        self.hop_size = int(sr * frame_period * 0.001)
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.n_mels = n_mels
        self.exclude_0th_component_in_LSD = exclude_0th_component_in_LSD
        self.exclude_0th_component_in_MCD = exclude_0th_component_in_MCD

        # For calculating F0-RMSE, LSD and MCD
        self.n_frames = [0 for _ in range(3)]
        self.means = [0.0 for _ in range(3)]

        # For calculating V/UV decision error
        self.voiced_threshold = 0  # Threshold used to decide V/UV
        self.n_uv_error_frame = 0  # Initialize number of V/UV errored frames
        self.n_all_frame = 0  # Initialize number of all frames

    def online_mean(self, n, mean, x):
        new_mean = (n * mean + x) / (n + 1)
        mean = new_mean
        n += 1
        return n, new_mean

    def calc_latent_mse(self, z1, z2):
        diff_latent_space = (z1 - z2) ** 2
        latent_mse = np.mean(diff_latent_space.detach().numpy().copy())
        return latent_mse

    def append_data(self, z1, z2, f0_factor=1.0):
        # calculate scores
        scores = [
            self.calc_latent_mse(z1, z2),
        ]

        # calculate new means of scores
        self.n_frames[0], self.means[0] = self.online_mean(
            self.n_frames[0], self.means[0], scores[0]
        )

@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Calculate mse on {device}.")

    # load pre-trained model from checkpoint file
    if config.checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.out_dir,
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        checkpoint_path = config.checkpoint_path
    state_dict = torch.load(to_absolute_path(checkpoint_path), map_location="cpu")
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    model = hydra.utils.instantiate(config.generator)
    model.load_state_dict(state_dict["model"]["generator"])
    model.decoder.remove_weight_norm()
    model.eval().to(device)

    # check directory existence
    out_dir = to_absolute_path(os.path.join(config.out_dir, "wav", str(config.checkpoint_steps)))
    os.makedirs(out_dir, exist_ok=True)

    f0_factor = 1.0
    dataset = FeatDataset(
        stats=to_absolute_path(config.data.stats),
        feat_list=config.data.eval_feat,
        return_filename=True,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
        f0_factor=f0_factor,
    )
    
    obj_score = ObjectiveScore(
        sr=config.data.sample_rate,
        n_fft=2048,
        f0_floor=100.0,  # should be adjusted if f0_factor < 1.0
        f0_ceil=1000.0,  # should be adjusted if f0_factor > 1.0
    )

    file_cnt = 0
    means = obj_score.means
    with torch.no_grad(), tqdm(dataset, desc="[calc_mse]") as pbar:
        for idx, (feat_path, c, f0) in enumerate(pbar, 1):
            logger.info(f"Processing {feat_path}")
            spec_lengths = torch.LongTensor([c.shape[0]]).to(device)
            mel = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)  # (1, 80, T)
            f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)
            # outs: [mu, logvar, x_, s, z]
            outs = model(mel, spec_lengths, f0)
            if file_cnt % 4 == 0:
                _, z1 = outs[2], outs[4]
            elif file_cnt % 4 == 1:
                _, z2 = outs[2], outs[4]
            elif file_cnt % 4 == 2:
                _, z3 = outs[2], outs[4]
            else:
                _, z4 = outs[2], outs[4]
                # online calculation
                obj_score.append_data(z3, z4, f0_factor)
                logger.info(f"MSE of latent space: {means[0]}")
            file_cnt += 1
        print(f"Processed {file_cnt} files.")
        
        # report average RTF
        logger.info(f"Finished calculation of mse.")
        logger.info(f"MSE of latent space: {means[0]}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding Script for SiFi-GAN VAE.

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
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from sifigan.datasets import FeatDataset

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

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

    total_rtf = 0.0
    for f0_factor in config.f0_factors:
        dataset = FeatDataset(
            stats=to_absolute_path(config.data.stats),
            feat_list=config.data.eval_feat,
            return_filename=True,
            sample_rate=config.data.sample_rate,
            hop_size=config.data.hop_size,
            aux_feats=config.data.aux_feats,
            f0_factor=f0_factor,
        )
        logger.info(f"The number of features to be decoded = {len(dataset)}.")

        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for idx, (feat_path, c, f0, cf0) in enumerate(pbar, 1):
                spec_lengths = torch.LongTensor([c.shape[0]]).to(device)
                # mel: 80-dimensional log-melspectrogram
                mel = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)  # (1, 80, T)
                f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)

                # perform decoding
                start = time()
                # outs: [mu, logvar, x_, s]
                outs = model(mel, spec_lengths, f0)
                y = outs[2]
                rtf = (time() - start) / (y.size(-1) / config.data.sample_rate)
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                # save output signal as PCM 16 bit wav file
                utt_id = os.path.splitext(os.path.basename(feat_path))[0]
                save_path = os.path.join(out_dir, f"{utt_id}_f{f0_factor:.2f}.wav")
                y = y.view(-1).cpu().numpy()
                sf.write(save_path, y, config.data.sample_rate, "PCM_16")

                # save source signal as PCM 16 bit wav file
                if config.save_source:
                    save_path = save_path.replace(".wav", "_s.wav")
                    s = outs[3].view(-1).cpu().numpy()
                    s = s / np.max(np.abs(s))  # normalize
                    sf.write(save_path, s, config.data.sample_rate, "PCM_16")

            # report average RTF
            logger.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.4f}).")


if __name__ == "__main__":
    main()

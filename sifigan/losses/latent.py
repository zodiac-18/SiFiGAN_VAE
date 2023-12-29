# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""MSE loss of latent spaces modules."""

import sifigan.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LatentMSEloss(nn.Module):
    """Calculate MSE loss of latent spaces."""

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, z1, z2):
        if self.loss_type == "mse":
            diff_latent_space = (z1 - z2) ** 2
            latent_mse = np.mean(diff_latent_space.cpu().detach().numpy().copy())
        return latent_mse
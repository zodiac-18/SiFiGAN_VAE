# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""MSE loss of latent spaces modules."""

import sifigan.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class Latentloss(nn.Module):
    """Calculate distance loss of latent spaces."""

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, z1=None, z2=None):
        if self.loss_type == "mse":
            diff_latent_space = (z1 - z2) ** 2
            latent_loss = torch.mean(diff_latent_space)
        elif self.loss_type == "cossim":
            # 1に近いほど良い
            latent_loss = 1 - torch.mean(F.cosine_similarity(z1, z2, dim=-1))
        else:
            latent_loss = 0.0
        return latent_loss
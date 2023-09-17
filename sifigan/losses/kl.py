# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Source regularization loss modules."""

import sifigan.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel
from sifigan.layers import CheapTrick


class KLDivergenceLoss(nn.Module):
    """The KL divergence Loss of VITS Posterior Encoder."""

    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var):
        mu = mu.float()
        log_var = log_var.float()

        kl_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        print(kl_loss)
        return kl_loss
from logging import getLogger

import torch
import torch.nn as nn

from sifigan.models import SiFiGANGenerator
from sifigan.utils import WN

# A logger for this file
logger = getLogger(__name__)


# 線形スペクトログラムを入力とし、入力データからzを出力する
class VITSPosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_x_channels,  # 入力するxのスペクトログラムの周波数の次元
        out_z_channels,  # 出力するzのチャネル数
        hidden_channels,  # 隠れ層のチャネル数
        kernel_size,  # WN内のconv1dのカーネルサイズ
        dilation_rate,  # WN内におけるconv1dのdilationの数値
        n_resblock,  # WN内のResidual Blockの重ねる数
        gin_channels=0,
    ):  # Gated Information Network(GIN)のチャネル数(default=0)
        super(VITSPosteriorEncoder, self).__init__()
        self.in_x_channels = in_x_channels
        self.out_z_channels = out_z_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_resblock
        self.gin_channels = gin_channels

        # 入力スペクトログラムに対して前処理を行う層
        self.preprocess = nn.Conv1d(in_x_channels, hidden_channels, 1)
        # WNを用いて特徴量の抽出を行う
        self.encode = WN(hidden_channels, kernel_size, dilation_rate, n_resblock, gin_channels=gin_channels)
        # ガウス分布の平均と分散を生成する層
        self.projection = nn.Conv1d(hidden_channels, out_z_channels * 2, 1)

    def forward(self, x_spec, x_spec_lengths, g=None):
        # マスクの作成
        # スペクトログラムの3番目の次元(=時間軸)のサイズから最大フレーム数を取得
        max_length = x_spec.size(2)
        # フレームの時間的な位置情報
        progression = torch.arange(max_length, dtype=x_spec_lengths.dtype, device=x_spec_lengths.device)
        # スペクトログラムの各フレームに対してその時間的位置がそのスペクトログラムの長さ未満(=有効)であるかどうかを示すbool値のテンソル
        x_spec_mask = progression.unsqueeze(0) < x_spec_lengths.unsqueeze(1)
        x_spec_mask = torch.unsqueeze(x_spec_mask, 1).to(x_spec.dtype)
        # preprocess層で畳み込みし、maskを適用
        x_spec = self.preprocess(x_spec) * x_spec_mask
        # WNでエンコードして特徴量を抽出
        x_spec = self.encode(x_spec, x_spec_mask, g=g)
        # 特徴量をガウス分布の平均と対数分散に変換する(マスクを適用)
        stats = self.projection(x_spec) * x_spec_mask
        # statsから平均mと対数分散logsを分離する, out_channelsでzのチャネル数を指定
        m, logs = torch.split(stats, self.out_z_channels, dim=1)
        # reparameterization trick(z=μ+εσ)によりzを擬似サンプリング(マスクを適用)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_spec_mask
        return m, logs, z, x_spec_mask


class VAEGANVocoder(nn.Module):
    def __init__(
        self,
        encoder_params={
            "in_x_channels": 80,  # 入力するxのスペクトログラムの周波数の次元
            "out_z_channels": 43,  # 出力するzのチャネル数
            "hidden_channels": 192,  # 隠れ層のチャネル数
            "kernel_size": 7,  # WN内のconv1dのカーネルサイズ
            "dilation_rate": 0.1,  # WN内におけるconv1dのdilationの数値
            "n_resblock": 16,  # WN内のResidual Blockの重ねる数
            "gin_channels": 0,
        },  # e.g., VITS poseterior encoder params
        decoder_params={
            "in_z_channels": 43,
            "out_x_channels": 1,
            "channels": 512,
            "kernel_size": 7,
            "upsample_scales": (5, 4, 3, 2),
            "upsample_kernel_sizes": (10, 8, 6, 4),
            "source_network_params": {
                "resblock_kernel_size": 3,
                "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
                "use_additional_convs": True,
            },
            "filter_network_params": {
                "resblock_kernel_sizes": (3, 5, 7),
                "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                "use_additional_convs": False,
            },
            "sample_rate": 24000,
            "dense_factors": [0.2, 1, 3, 6],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "bias": True,
            "use_weight_norm": True,
        },
    ):
        super(VAEGANVocoder, self).__init__()
        self.encoder = VITSPosteriorEncoder(**encoder_params)
        self.decoder = SiFiGANGenerator(**decoder_params)

    def forward(self, mel, spec_length, f0):
        # encode mel to z
        mu, logvar, z, _ = self.encoder(mel, spec_length)

        # decode z to waveform
        x, s = self.decoder(z, f0)
        # x, s = self.decoder(mel, f0)

        return mu, logvar, x, s, z
        # return mu, logvar, x, s

class VAEGANVocoder_avg(nn.Module):
    def __init__(
        self,
        encoder_params={
            "in_x_channels": 80,  # 入力するxのスペクトログラムの周波数の次元
            "out_z_channels": 43,  # 出力するzのチャネル数
            "hidden_channels": 192,  # 隠れ層のチャネル数
            "kernel_size": 7,  # WN内のconv1dのカーネルサイズ
            "dilation_rate": 0.1,  # WN内におけるconv1dのdilationの数値
            "n_resblock": 16,  # WN内のResidual Blockの重ねる数
            "gin_channels": 0,
        },  # e.g., VITS poseterior encoder params
        decoder_params={
            "in_z_channels": 43,
            "out_x_channels": 1,
            "channels": 512,
            "kernel_size": 7,
            "upsample_scales": (5, 4, 3, 2),
            "upsample_kernel_sizes": (10, 8, 6, 4),
            "source_network_params": {
                "resblock_kernel_size": 3,
                "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
                "use_additional_convs": True,
            },
            "filter_network_params": {
                "resblock_kernel_sizes": (3, 5, 7),
                "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                "use_additional_convs": False,
            },
            "sample_rate": 24000,
            "dense_factors": [0.2, 1, 3, 6],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "bias": True,
            "use_weight_norm": True,
        },
    ):
        super(VAEGANVocoder_avg, self).__init__()
        self.encoder = VITSPosteriorEncoder(**encoder_params)
        self.decoder = SiFiGANGenerator(**decoder_params)

    def forward(self, mel_source, mel_trans_1, mel_trans_2, spec_length, f0):
        # 音源, 変換音声1, 変換音声2の場合のみを想定
        # encode mel to z
        mu_source, logvar_source, z_source, x_spec_mask_source = self.encoder(mel_source, spec_length)
        mu_trans_1, logvar_trans_1, z_trans_1, _ = self.encoder(mel_trans_1, spec_length)
        mu_trans_2, logvar_trans_2, z_trans_2, _ = self.encoder(mel_trans_2, spec_length)
        mu_list, logvar_list, z_list = [mu_source, mu_trans_1, mu_trans_2], [logvar_source, logvar_trans_1, logvar_trans_2], [z_source, z_trans_1, z_trans_2]
        m, var = (mu_source + mu_trans_1 + mu_trans_2) / 3, (logvar_source + logvar_trans_1 + logvar_trans_2) / 3
        z = (m + torch.randn_like(m) * torch.exp(var)) * x_spec_mask_source
        # decode z to waveform
        x, s = self.decoder(z, f0)
        # x, s = self.decoder(mel, f0)

        return mu_list, logvar_list, x, s, z_list
        # return mu, logvar, x, s
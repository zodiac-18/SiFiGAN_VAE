import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import WN

# A logger for this file
logger = getLogger(__name__)

# 線形スペクトログラムを入力とし、入力データからzを出力する
class VITSPosteriorEncoder(nn.Module):
  def __init__(self,
      in_x_channels, #入力するxのスペクトログラムの周波数の次元
      out_z_channels, #出力するzのチャネル数
      hidden_channels, #隠れ層のチャネル数
      kernel_size, #WN内のconv1dのカーネルサイズ
      dilation_rate, #WN内におけるconv1dのdilationの数値
      n_resblock, #WN内のResidual Blockの重ねる数
      gin_channels=0): #Gated Information Network(GIN)のチャネル数(default=0)
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
    x_spec_mask = (progression.unsqueeze(0) < x_spec_lengths.unsqueeze(1))
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


def pd_indexing1d(x, d, dilation):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.

    Returns:
        Tensor: Past output tensor (B, C, T).
        Tensor: Center element tensor (B, C, T).
        Tensor: Future output tensor (B, C, T).

    """
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)

    # get past index (assume reflect padding)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)

    # get future index (assume reflect padding)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T) - 1
    idxF = (batch_index, ch_index, idxF)

    return x[idxP], x, x[idxF]


class AdaptiveConv1d(nn.Module):
    """F0 adaptive conv1d module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        bias=True,
    ):
        """Initialize AdaptiveConv2d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution layer.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super().__init__()
        self.dilation = dilation
        # NOTE: currently only kernel_size = 3 is supported.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=bias if i == 0 else False,
                )
                for i in range(kernel_size)
            ]
        )

    def forward(self, x, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            d (Tensor): F0-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        out = 0.0
        xs = pd_indexing1d(x, d, self.dilation)
        for x, f in zip(xs, self.convs):
            out = out + f(x)
        return out


class ResidualBlock(nn.Module):
    """Residual block module in SiFi-GAN generator."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        use_adaptive_convs=False,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        bias=True,
    ):
        """Initialize HiFiGAN's ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_adaptive_convs (bool): Whether to use F0 adaptive convolution layers.
            use_additional_convs (bool): Whether to use additional convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        self.dilations = dilations
        self.use_adaptive_convs = use_adaptive_convs
        self.use_additional_convs = use_additional_convs
        self.nonlinears1 = nn.ModuleList()
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.nonlinears2 = nn.ModuleList()
            self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.nonlinears1 += [getattr(nn, nonlinear_activation)(**nonlinear_activation_params)]
            self.convs1 += [
                AdaptiveConv1d(
                    channels,
                    channels,
                    dilation=dilation,
                    bias=bias,
                )
                if use_adaptive_convs
                else nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    padding=(kernel_size - 1) // 2 * dilation,
                )
            ]

            if use_additional_convs:
                self.nonlinears2 += [getattr(nn, nonlinear_activation)(**nonlinear_activation_params)]
                self.convs2 += [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        bias=bias,
                        padding=(kernel_size - 1) // 2,
                    )
                ]

    def forward(self, x, d=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            d (Tensor): F0-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.dilations)):
            xt = self.nonlinears1[idx](x)
            if self.use_adaptive_convs:
                xt = self.convs1[idx](xt, d)
            else:
                xt = self.convs1[idx](xt)
            if self.use_additional_convs:
                xt = self.nonlinears2[idx](xt)
                xt = self.convs2[idx](xt)
            x = xt + x

        return x


class SiFiGANGenerator(nn.Module):
    """SiFi-GAN generator module."""

    def __init__(
        self,
        in_z_channels,
        out_x_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(5, 4, 3, 2),
        upsample_kernel_sizes=(10, 8, 6, 4),
        source_network_params={
            "resblock_kernel_size": 3,
            "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
            "use_additional_convs": True,
        },
        filter_network_params={
            "resblock_kernel_sizes": (3, 5, 7),
            "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            "use_additional_convs": False,
        },
        sample_rate=24000,
        dense_factors=[0.2, 1, 3, 6],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        bias=True,
        use_weight_norm=True,
    ):
        """Initialize SiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            source_network_params (dict): Parameters for source-network.
            filter_network_params (dict): Parameters for filter-network.
            sample_rate (int): Sampling frequency of output waveform.
            dense_factors (list): The base numbers of taps in one cycle.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        self.in_channels = in_z_channels
        self.upsample_scales = upsample_scales
        self.num_upsamples = len(upsample_kernel_sizes)
        self.hop_length = np.prod(upsample_scales)
        self.sample_rate = sample_rate
        self.dense_factors = dense_factors

        # define input conv
        self.input_conv = nn.Conv1d(
            in_z_channels,
            channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        # define residual convs
        self.source_network_params = source_network_params
        self.filter_network_params = filter_network_params
        self.sn = nn.ModuleDict()
        self.fn = nn.ModuleDict()
        self.sn["upsamples"] = nn.ModuleList()
        self.fn["upsamples"] = nn.ModuleList()
        self.sn["blocks"] = nn.ModuleList()
        self.fn["blocks"] = nn.ModuleList()
        for i in range(self.num_upsamples):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.sn["upsamples"] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                        bias=bias,
                    ),
                )
            ]
            self.fn["upsamples"] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                        bias=bias,
                    ),
                )
            ]
            self.sn["blocks"] += [
                ResidualBlock(
                    kernel_size=source_network_params["resblock_kernel_size"],
                    channels=channels // (2 ** (i + 1)),
                    dilations=source_network_params["resblock_dilations"][i],
                    use_adaptive_convs=True,
                    use_additional_convs=source_network_params["use_additional_convs"],
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    bias=bias,
                )
            ]
            for j in range(len(filter_network_params["resblock_kernel_sizes"])):
                self.fn["blocks"] += [
                    ResidualBlock(
                        kernel_size=filter_network_params["resblock_kernel_sizes"][j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=filter_network_params["resblock_dilations"][j],
                        use_adaptive_convs=False,
                        use_additional_convs=filter_network_params["use_additional_convs"],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        bias=bias,
                    )
                ]

        # define sine waveform embedding layer
        self.emb = nn.Conv1d(
            1,
            channels // (2 ** len(upsample_kernel_sizes)),
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        # define down-sampling layers
        self.sn["downsamples"] = nn.ModuleList()
        for i in reversed(range(1, self.num_upsamples)):
            self.sn["downsamples"] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.Conv1d(
                        channels // (2 ** (i + 1)),
                        channels // (2**i),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] - (upsample_kernel_sizes[i] % 2 == 0),
                        bias=bias,
                    ),
                )
            ]

        self.fn["downsamples"] = nn.ModuleList()
        for i in reversed(range(1, self.num_upsamples)):
            self.fn["downsamples"] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.Conv1d(
                        channels // (2 ** (i + 1)),
                        channels // (2**i),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] - (upsample_kernel_sizes[i] % 2 == 0),
                        bias=bias,
                    ),
                )
            ]

        # define output conv
        self.output_conv = nn.Sequential(
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            nn.Conv1d(
                channels // (2 ** len(upsample_kernel_sizes)),
                out_x_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, f0):
        """Calculate forward propagation.

        Args:
            c (Tensor): Acoustic feature (B, in_channels, T).
            f0 (Tensor): F0 sequence (B, 1, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T * hop_length).

        """

        # generate periodic waveform embeddings
        sine = self.generate_sine(f0)
        x = self.emb(sine.float())
        embs = [x]
        for i in range(self.num_upsamples - 1):
            x = self.sn["downsamples"][i](x)
            embs += [x]

        # generate f0-dependent dilation factors
        d = self.dilated_factor(f0)

        # apply input conv
        c = self.input_conv(c)

        # apply source-network residual convs
        s = c
        for i in range(self.num_upsamples):
            s = self.sn["upsamples"][i](s) + embs[-i - 1]
            s = self.sn["blocks"][i](s, d[i])

        # apply output conv
        s_ = self.output_conv(s)

        # generate source excitation representation embeddings
        embs = [s]
        for i in range(self.num_upsamples - 1):
            s = self.fn["downsamples"][i](s)
            embs += [s]

        # apply filter-network residual convs
        num_blocks = len(self.filter_network_params["resblock_kernel_sizes"])
        for i in range(self.num_upsamples):
            c = self.fn["upsamples"][i](c) + embs[-i - 1]
            cs = 0.0
            for j in range(num_blocks):
                cs += self.fn["blocks"][i * num_blocks + j](c)
            c = cs / num_blocks

        # apply output conv
        x = self.output_conv(c)

        return x, s_

    # フレームレベルのF0を受け取ってサンプルレベルの正弦波を出力する関数
    def generate_sine(self, f0):
        """Sinusoidal waveform generator

        Args:
            f0 (Tensor): F0 sequence (B, 1, T)

        Return:
            sine (Tensor):
                Sinusoidal waveform (B, 1, T * hop_length)

        """
        device = f0.device
        B, _, T = f0.size()
        if torch.all(f0 == 0.0):
            # 全て無声区間ならノイズ
            return torch.randn((B, 1, T * self.hop_length), device=device)
        else:
            f0 = torch.nn.functional.interpolate(
                f0.to(torch.float64), size=self.hop_length * T, mode="nearest"
            )  # (B, 1, T * hop_length)
            vuv = f0 > 0
            phase = 2.0 * np.pi * torch.cumsum(f0 / self.sample_rate, dim=-1)
            sine = 0.1 * vuv * torch.cos(phase)
            noise = 0.03 * torch.randn(sine.size(), device=device)
            sine = sine + noise
            return sine

    def dilated_factor(self, f0):
        """F0-dependent dilated factor

        Args:
            f0 (Tensor): F0 sequence (B, 1, T)

        Return:
            dilated_factors (List):
                F0-dependent dilated factors [(B, 1, *) x num_upsamples]

        """
        initial_sample_rate = self.sample_rate // np.prod(
            self.upsample_scales
        )  # e.g. 24000 [Hz] / prod([8, 5, 3, 2]) = 100 [Hz]
        dilated_factors = []
        for dense_factor, upsample_scale in zip(self.dense_factors, np.cumprod(self.upsample_scales)):
            sample_rate = initial_sample_rate * upsample_scale  # e.g., 100 [Hz] * (8 or 40 or 120 or 240)
            dilated_factor = torch.ones_like(f0)
            # The lower the sampling rate is, the lower f0 can be handled. So the dense_factor is
            # configured to handle higher f0 values at a given lower sampling rate.
            dilated_factor[f0 > 0] = sample_rate / (dense_factor * f0[f0 > 0])
            dilated_factor = torch.repeat_interleave(dilated_factor, upsample_scale, dim=-1)
            dilated_factors += [dilated_factor]

        return dilated_factors

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logger.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f"Weight norm is removed from {m}.")
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class VAEGANVocoder(nn.Module):
    def __init__(
        self,
        encoder_params={"in_x_channels":80, #入力するxのスペクトログラムの周波数の次元
                        "out_z_channels":192, #出力するzのチャネル数
                        "hidden_channels":256, #隠れ層のチャネル数
                        "kernel_size":7, #WN内のconv1dのカーネルサイズ
                        "dilation_rate":0.1, #WN内におけるconv1dのdilationの数値
                        "n_resblock":16, #WN内のResidual Blockの重ねる数
                        "gin_channels":0},  # e.g., VITS poseterior encoder params
        decoder_params={"in_z_channels":192,
                        "out_x_channels":1,
                        "channels":512,
                        "kernel_size":7,
                        "upsample_scales":(5, 4, 3, 2),
                        "upsample_kernel_sizes":(10, 8, 6, 4),
                        "source_network_params":{
                            "resblock_kernel_size": 3,
                            "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
                            "use_additional_convs": True,
                        },
                        "filter_network_params":{
                            "resblock_kernel_sizes": (3, 5, 7),
                            "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                            "use_additional_convs": False,
                        },
                        "sample_rate":24000,
                        "dense_factors":[0.2, 1, 3, 6],
                        "nonlinear_activation":"LeakyReLU",
                        "nonlinear_activation_params":{"negative_slope": 0.1},
                        "bias":True,
                        "use_weight_norm":True,}
    ):
        super(VAEGANVocoder, self).__init__()
        self.encoder = VITSPosteriorEncoder(**encoder_params)
        self.decoder = SiFiGANGenerator(**decoder_params)

    def forward(self, mel, spec_length, f0):
        # encode mel to z
        mu, logvar, z, _ = self.encoder(mel, spec_length)

        # decode z to waveform
        x, s = self.decoder(z, f0)

        return mu, logvar, x, s
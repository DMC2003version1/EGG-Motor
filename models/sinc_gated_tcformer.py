"""Parameter-efficient Sinc-Gated TCFormer.

It retains TCFormer's spatial CNN, GQA encoder, and TCN head.  Only the
unconstrained temporal kernels are replaced by learnable band-pass Sinc filters,
and CNN/Transformer features are fused with a lightweight residual gate.
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from .classification_module import ClassificationModule
from .tcformer import MultiKernelConvBlock, TCFormerModule, TCNHead, _TransformerBlock, _build_rotary_cache
from utils.latency import measure_latency


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-4, 1 - 1e-4)
    return torch.log(value / (1 - value))


class SincTemporalConv(nn.Module):
    """A bank of learnable, physically constrained 1-D band-pass filters."""
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: float = 250.0,
                 min_low_hz: float = 1.0, min_band_hz: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sample_rate = float(sample_rate)
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        nyquist = self.sample_rate / 2.0

        # Spread initial filters over the MI-relevant mu/beta range. The learned
        # cut-offs remain valid band-pass filters throughout training.
        initial_low = torch.linspace(4.0, 28.0, out_channels)
        low_scale = nyquist - min_low_hz - min_band_hz
        self.low_param = nn.Parameter(_logit((initial_low - min_low_hz) / low_scale))
        initial_band = torch.full((out_channels,), 6.0)
        band_scale = nyquist - initial_low - min_band_hz
        self.band_param = nn.Parameter(_logit((initial_band - min_band_hz) / band_scale))

    def forward(self, x: Tensor) -> Tensor:  # (B, 1, C, T)
        device, dtype = x.device, x.dtype
        nyquist = self.sample_rate / 2.0
        low_scale = nyquist - self.min_low_hz - self.min_band_hz
        low = self.min_low_hz + low_scale * torch.sigmoid(self.low_param)
        band_scale = nyquist - low - self.min_band_hz
        high = low + self.min_band_hz + band_scale * torch.sigmoid(self.band_param)

        time = (torch.arange(self.kernel_size, device=device, dtype=dtype)
                - (self.kernel_size - 1) / 2) / self.sample_rate
        low = low.to(dtype).unsqueeze(1)
        high = high.to(dtype).unsqueeze(1)
        band_pass = 2 * high * torch.sinc(2 * high * time) - 2 * low * torch.sinc(2 * low * time)
        window = torch.hamming_window(self.kernel_size, periodic=False, device=device, dtype=dtype)
        filters = band_pass * window
        filters = filters / filters.abs().sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        filters = filters.view(-1, 1, 1, self.kernel_size)

        left = self.kernel_size // 2 - (1 if self.kernel_size % 2 == 0 else 0)
        right = self.kernel_size // 2
        return F.conv2d(F.pad(x, (left, right, 0, 0)), filters)


class SincMultiKernelConvBlock(MultiKernelConvBlock):
    """Original TCFormer front-end with Sinc filters in the first temporal stage."""
    def __init__(self, n_channels: int, temp_kernel_lengths=(20, 32, 64), F1: int = 32,
                 sample_rate: float = 250.0, **kwargs):
        super().__init__(n_channels=n_channels, temp_kernel_lengths=temp_kernel_lengths, F1=F1, **kwargs)
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                SincTemporalConv(F1, kernel_size, sample_rate=sample_rate),
                nn.BatchNorm2d(F1),
            )
            for kernel_size in temp_kernel_lengths
        ])


class SincGatedTCFormerModule(TCFormerModule):
    def __init__(self, n_channels: int, n_classes: int, sample_rate: float = 250.0, **kwargs):
        super().__init__(n_channels=n_channels, n_classes=n_classes, **kwargs)
        temp_kernel_lengths = kwargs.get("temp_kernel_lengths", (16, 32, 64))
        d_group = kwargs.get("d_group", 16)
        self.conv_block = SincMultiKernelConvBlock(
            n_channels=n_channels,
            temp_kernel_lengths=temp_kernel_lengths,
            F1=kwargs.get("F1", 16),
            sample_rate=sample_rate,
            D=kwargs.get("D", 2),
            pool_length_1=kwargs.get("pool_length_1", 8),
            pool_length_2=kwargs.get("pool_length_2", 7),
            dropout=kwargs.get("dropout_conv", 0.3),
            d_group=d_group,
            use_group_attn=kwargs.get("use_group_attn", True),
        )

        # Project the compact Transformer representation back to CNN width. A
        # residual gate lets the model retain robust local CNN features when the
        # global representation is unhelpful for a particular trial.
        self.trans_to_conv = nn.Sequential(
            nn.Conv1d(d_group, self.d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )
        self.fusion_gate = nn.Conv1d(2 * self.d_model, self.d_model, kernel_size=1, bias=True)
        # The original concatenation gives the TCN 64 channels in four groups.
        # Fused features retain three temporal groups at 48 channels instead.
        self.tcn_head = TCNHead(self.d_model, len(temp_kernel_lengths),
                                kwargs.get("tcn_depth", 2), kwargs.get("kernel_length_tcn", 4),
                                kwargs.get("dropout_tcn", 0.3), n_classes)

    def forward(self, x: Tensor) -> Tensor:
        conv_features = self.conv_block(x)  # (B, d_model, T)
        _, _, token_length = conv_features.shape
        tokens = self.rearrange(self.mix(conv_features))
        cos, sin = self._rotary_cache(token_length, tokens.device)
        for block in self.transformer:
            tokens = block(tokens, cos, sin)
        trans_features = self.reduce(tokens)             # (B, d_group, T)
        trans_features = self.trans_to_conv(trans_features)  # (B, d_model, T)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((conv_features, trans_features), dim=1)))
        fused_features = conv_features + gate * trans_features
        return self.tcn_head(fused_features)


class SincGatedTCFormer(ClassificationModule):
    def __init__(self, n_channels: int, n_classes: int, sample_rate: float = 250.0, **kwargs):
        architecture_keys = {
            "F1", "temp_kernel_lengths", "pool_length_1", "pool_length_2", "D",
            "dropout_conv", "d_group", "tcn_depth", "kernel_length_tcn",
            "dropout_tcn", "use_group_attn", "q_heads", "kv_heads",
            "trans_dropout", "drop_path_max", "trans_depth",
        }
        module_kwargs = {key: value for key, value in kwargs.items() if key in architecture_keys}
        model = SincGatedTCFormerModule(n_channels=n_channels, n_classes=n_classes,
                                        sample_rate=sample_rate, **module_kwargs)
        super().__init__(model, n_classes, **kwargs)

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(SincGatedTCFormer(22, 4), input_shape, device, warmup, runs)

"""
Frequency-aware MoE + subject-adaptive TCFormer.

This experimental variant keeps the subject-conditioned HyperAdapters from
SubjectAdaptiveTCFormer and adds a trial-wise spectral mixture-of-experts in
front of the TCFormer feature extractor.
"""

import torch
from torch import nn, Tensor

from .classification_module import ClassificationModule
from .subject_adaptive_tcformer import SubjectAdaptiveTCFormerModule
from utils.latency import measure_latency


class SpectralGate(nn.Module):
    """
    Builds expert weights from label-free FFT band-power descriptors.

    The gate sees coarse mu/beta/broad-band power statistics and outputs a
    softmax distribution over temporal experts for each trial.
    """
    def __init__(
        self,
        n_experts: int,
        sfreq: float = 250.0,
        bands=((8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (4.0, 40.0)),
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.sfreq = float(sfreq)
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.net = nn.Sequential(
            nn.LayerNorm(len(self.bands)),
            nn.Linear(len(self.bands), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / self.sfreq).to(x.device)
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().pow(2).mean(dim=1)

        band_features = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                band_power = power[:, mask].mean(dim=-1)
            else:
                band_power = power.new_zeros(power.size(0))
            band_features.append(torch.log(band_power.clamp_min(1e-6)))

        features = torch.stack(band_features, dim=-1)
        return torch.softmax(self.net(features), dim=-1)


class TemporalFrequencyExpert(nn.Module):
    """
    Lightweight depthwise-separable temporal expert over raw EEG channels.
    """
    def __init__(self, n_channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FrequencyAwareMoE(nn.Module):
    """
    Trial-adaptive spectral expert mixer.

    Experts operate on the raw EEG sequence and are mixed with FFT-derived gate
    weights. A zero-initialized residual scale lets training start from the
    original input path and learn the spectral correction gradually.
    """
    def __init__(
        self,
        n_channels: int,
        sfreq: float = 250.0,
        expert_kernel_sizes=(63, 47, 31, 15),
        dropout: float = 0.1,
        gate_hidden_dim: int = 32,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            TemporalFrequencyExpert(n_channels, int(kernel), dropout)
            for kernel in expert_kernel_sizes
        ])
        self.gate = SpectralGate(
            n_experts=len(self.experts),
            sfreq=sfreq,
            hidden_dim=gate_hidden_dim,
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        weights = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_outputs * weights[:, :, None, None]).sum(dim=1)
        return x + self.residual_scale * mixed


class FrequencyMoESubjectAdaptiveTCFormerModule(SubjectAdaptiveTCFormerModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        sfreq: float = 250.0,
        moe_kernel_sizes=(63, 47, 31, 15),
        moe_dropout: float = 0.1,
        moe_gate_hidden_dim: int = 32,
        **kwargs,
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            **kwargs,
        )
        self.frequency_moe = FrequencyAwareMoE(
            n_channels=n_channels,
            sfreq=sfreq,
            expert_kernel_sizes=moe_kernel_sizes,
            dropout=moe_dropout,
            gate_hidden_dim=moe_gate_hidden_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.frequency_moe(x)
        return super().forward(x)


class FrequencyMoESubjectAdaptiveTCFormer(ClassificationModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        F1: int = 16,
        temp_kernel_lengths: tuple = (16, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.3,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        q_heads: int = 8,
        kv_heads: int = 4,
        trans_depth: int = 5,
        trans_dropout: float = 0.4,
        subject_emb_dim: int = 32,
        adapter_bottleneck_ratio: int = 4,
        adapter_dropout: float = 0.1,
        sfreq: float = 250.0,
        moe_kernel_sizes: tuple = (63, 47, 31, 15),
        moe_dropout: float = 0.1,
        moe_gate_hidden_dim: int = 32,
        **kwargs,
    ):
        model = FrequencyMoESubjectAdaptiveTCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            trans_dropout=trans_dropout,
            subject_emb_dim=subject_emb_dim,
            adapter_bottleneck_ratio=adapter_bottleneck_ratio,
            adapter_dropout=adapter_dropout,
            sfreq=sfreq,
            moe_kernel_sizes=moe_kernel_sizes,
            moe_dropout=moe_dropout,
            moe_gate_hidden_dim=moe_gate_hidden_dim,
        )
        super().__init__(model, n_classes, **kwargs)

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(
            FrequencyMoESubjectAdaptiveTCFormer(22, 4),
            input_shape,
            device,
            warmup,
            runs,
        )

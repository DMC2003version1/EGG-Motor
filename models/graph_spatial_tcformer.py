"""Topology-aware spatial graph variant of TCFormer.

The dual spatial encoder combines depth-wise convolution with topology-aware
graph attention, then uses learned temporal attention pooling before GQA.
"""
import torch
from torch import nn
from einops.layers.torch import Rearrange

from .classification_module import ClassificationModule
from .tcformer import TCFormerModule, MultiKernelConvBlock
from utils.weight_initialization import glorot_weight_zero_bias
from utils.latency import measure_latency


# BCI Competition IV-2a's 22 EEG electrodes in the dataset order.  The values
# are approximate 10-20/10-10 scalp coordinates; only neighbourhood matters.
_BCIC2A_COORDS = [
    (0, 3), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
    (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
    (-1, -3), (0, -3), (1, -3), (0, -4),
]


class TopologyGraphSpatialEncoder(nn.Module):
    """Masked, dynamic graph attention over EEG electrodes at every time step."""
    def __init__(self, n_channels, n_features, dropout=0.1, neighbour_radius=2.1):
        super().__init__()
        self.q = nn.Linear(n_features, n_features, bias=False)
        self.k = nn.Linear(n_features, n_features, bias=False)
        self.v = nn.Linear(n_features, n_features, bias=False)
        self.out = nn.Linear(n_features, n_features, bias=False)
        self.norm = nn.LayerNorm(n_features)
        self.drop = nn.Dropout(dropout)
        if n_channels == 22:
            coords = torch.tensor(_BCIC2A_COORDS, dtype=torch.float32)
            distance = torch.cdist(coords, coords)
            adjacency = distance <= neighbour_radius
        else:
            # No electrode names are available in the generic datamodules. Keep
            # self edges and immediate indexed neighbours rather than pretending
            # arbitrary channel order is a physical scalp map.
            indices = torch.arange(n_channels)
            adjacency = (indices[:, None] - indices[None, :]).abs() <= 1
        adjacency.fill_diagonal_(True)
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.scale = n_features ** -0.5

    def forward(self, x):  # x: (B, features, electrodes, time)
        nodes = x.permute(0, 3, 2, 1)  # (B, T, C, F)
        q, k, v = self.q(nodes), self.k(nodes), self.v(nodes)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.adjacency[None, None], float("-inf"))
        attention = self.drop(scores.softmax(dim=-1))
        updated = self.out(attention @ v)
        nodes = self.norm(nodes + updated)
        # Attention pooling retains one spatial representation at each time step.
        return nodes.mean(dim=2).permute(0, 2, 1).unsqueeze(2)


class LocalTemporalAttentionPool(nn.Module):
    """Learn a separate importance distribution inside each pooling window."""
    def __init__(self, n_features, window_size):
        super().__init__()
        self.window_size = window_size
        self.score = nn.Conv2d(n_features, n_features, (1, 1), groups=n_features)

    def forward(self, x):  # (B, F, 1, T)
        usable = (x.shape[-1] // self.window_size) * self.window_size
        if usable == 0:
            return x
        x = x[..., :usable]
        scores = self.score(x)
        shape = (*x.shape[:-1], usable // self.window_size, self.window_size)
        values = x.reshape(shape)
        weights = scores.reshape(shape).softmax(dim=-1)
        return (weights * values).sum(dim=-1)


class GraphMultiKernelConvBlock(MultiKernelConvBlock):
    """Parallel convolution/graph spatial paths with gated fusion and token pooling."""
    def __init__(self, n_channels, *args, graph_dropout=0.1, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        n_groups = len(self.temporal_convs)
        temporal_features = self.temporal_convs[0][1].out_channels * n_groups
        # Retain TCFormer's spatial convolution as one path.
        self.spatial_conv = self.channel_DW_conv
        self.graph_spatial = TopologyGraphSpatialEncoder(
            n_channels, temporal_features, dropout=graph_dropout
        )
        del self.channel_DW_conv
        F2 = temporal_features * kwargs.get("D", 2)
        self.graph_expand = nn.Sequential(
            nn.Conv2d(temporal_features, F2, (1, 1), bias=False, groups=temporal_features),
            nn.BatchNorm2d(F2), nn.ELU(),
        )
        self.fusion_gate = nn.Conv2d(2 * F2, F2, (1, 1), bias=True)
        self.pool1 = LocalTemporalAttentionPool(F2, kwargs.get("pool_length_1", 8))
        self.pool2 = LocalTemporalAttentionPool(self.d_model, kwargs.get("pool_length_2", 7))
        glorot_weight_zero_bias(self.graph_expand)

    def forward(self, x):
        x = self.rearrange(x)
        temporal = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        conv_features = self.spatial_conv(temporal)
        graph_features = self.graph_expand(self.graph_spatial(temporal))
        gate = torch.sigmoid(self.fusion_gate(torch.cat((conv_features, graph_features), dim=1)))
        x = gate * conv_features + (1.0 - gate) * graph_features
        x = self.drop1(self.pool1(x))
        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            x = x + self.group_attn(x)
        return self.drop2(self.pool2(x)).squeeze(2)


class GraphSpatialTCFormerModule(TCFormerModule):
    def __init__(self, n_channels, n_classes, graph_dropout=0.1, **kwargs):
        super().__init__(n_channels=n_channels, n_classes=n_classes, **kwargs)
        self.conv_block = GraphMultiKernelConvBlock(
            n_channels=n_channels,
            temp_kernel_lengths=kwargs.get("temp_kernel_lengths", (16, 32, 64)),
            F1=kwargs.get("F1", 16), D=kwargs.get("D", 2),
            pool_length_1=kwargs.get("pool_length_1", 8),
            pool_length_2=kwargs.get("pool_length_2", 7),
            dropout=kwargs.get("dropout_conv", 0.3),
            d_group=kwargs.get("d_group", 16),
            use_group_attn=kwargs.get("use_group_attn", True),
            graph_dropout=graph_dropout,
        )


class GraphSpatialTCFormer(ClassificationModule):
    def __init__(self, n_channels, n_classes, graph_dropout=0.1, **kwargs):
        # ClassificationModule consumes training arguments (lr, optimizer,
        # warmup, ...); TCFormerModule must receive architecture arguments only.
        architecture_keys = {
            "F1", "temp_kernel_lengths", "pool_length_1", "pool_length_2", "D",
            "dropout_conv", "d_group", "tcn_depth", "kernel_length_tcn",
            "dropout_tcn", "use_group_attn", "q_heads", "kv_heads",
            "trans_dropout", "drop_path_max", "trans_depth",
        }
        module_kwargs = {key: value for key, value in kwargs.items() if key in architecture_keys}
        model = GraphSpatialTCFormerModule(
            n_channels=n_channels, n_classes=n_classes, graph_dropout=graph_dropout, **module_kwargs
        )
        super().__init__(model, n_classes, **kwargs)

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(GraphSpatialTCFormer(22, 4), input_shape, device, warmup, runs)

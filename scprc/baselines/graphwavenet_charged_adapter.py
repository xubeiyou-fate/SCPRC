"""Variable-node Graph WaveNet adapter for the CHARGED LOCO protocol.

The official Graph WaveNet implementation assumes that its learned adaptive
node embeddings have one fixed node count.  CHARGED cities have different
station sets, so this adapter keeps the temporal/gated/diffusion backbone but
turns off the node-ID adaptive adjacency.  Each city supplies a deterministic
distance support with the same top-k Gaussian rule.  The support is stored as
an edge list and propagated with gather, avoiding an O(N^2) dense einsum for
AMS/SZH while preserving the upstream diffusion orientation.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


_SOURCE = Path(__file__).resolve().parents[1] / "vendor" / "Graph-WaveNet" / "model.py"
_SPEC = importlib.util.spec_from_file_location("charged_pinned_graph_wavenet", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load pinned Graph WaveNet source: {_SOURCE}")
_OFFICIAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OFFICIAL)


@dataclass(frozen=True)
class FixedSupport:
    """Sparse ``A.T`` operator for the upstream ``x @ A`` diffusion."""

    operator: torch.Tensor

    @property
    def nodes(self) -> int:
        return int(self.operator.shape[0])

    def to(self, device: torch.device | str) -> "FixedSupport":
        if self.operator.device == torch.device(device):
            return self
        return FixedSupport(self.operator.to(device).coalesce())


def distance_supports(distance: np.ndarray, *, top_k: int = 8,
                      sigma: float | None = None) -> tuple[FixedSupport, FixedSupport]:
    """Build upstream-style double transitions from a symmetrized distance KNN.

    For each station, connect its ``top_k`` nearest stations, symmetrize the
    union, add self loops, and assign Gaussian weights. The two row-normalized
    supports correspond to ``A`` and ``A.T`` as in Graph WaveNet. Because the
    underlying distance graph is symmetric, these supports coincide, just as
    double transition does for any symmetric adjacency. This uses only static
    geometry, never time-series observations.
    """
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("distance must be a finite nonnegative square matrix")
    n = int(d.shape[0])
    if n == 0:
        raise ValueError("distance cannot be empty")
    k = min(max(1, int(top_k)), max(1, n - 1))
    positive = d[d > 0]
    auto_sigma = float(np.median(positive)) if positive.size else 1.0
    scale = float(sigma if sigma is not None else auto_sigma)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    # Exclude the diagonal while selecting neighbors, then add it explicitly.
    selection = d.copy()
    np.fill_diagonal(selection, np.inf)
    nearest = np.argpartition(selection, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[:, None], nearest] = True
    mask |= mask.T
    np.fill_diagonal(mask, True)
    raw = np.exp(-np.square(d / scale)) * mask
    def make_support(adjacency: np.ndarray) -> FixedSupport:
        row_sums = np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12)
        transition = adjacency / row_sums
        # Upstream computes x @ transition, equivalent to sparse
        # transition.T @ x after moving node to the leading dimension.
        transposed = transition.T
        indices = np.vstack(np.nonzero(transposed)).astype(np.int64)
        values = transposed[indices[0], indices[1]].astype(np.float32)
        operator = torch.sparse_coo_tensor(
            torch.as_tensor(indices, dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32),
            size=(n, n), dtype=torch.float32,
        ).coalesce()
        return FixedSupport(operator)

    return make_support(raw), make_support(raw.T)


class _FixedNConv(nn.Module):
    """Upstream nconv replacement supporting edge-list supports."""

    def forward(self, x: torch.Tensor, support: FixedSupport | torch.Tensor) -> torch.Tensor:
        if isinstance(support, FixedSupport):
            if x.ndim != 4 or x.shape[2] != support.nodes:
                raise ValueError(
                    f"support has {support.nodes} nodes but x is {tuple(x.shape)}"
                )
            b, c, n, t = x.shape
            matrix = x.permute(2, 0, 1, 3).reshape(n, b * c * t)
            operator = support.operator
            if operator.device != x.device or operator.dtype != x.dtype:
                operator = operator.to(device=x.device, dtype=x.dtype).coalesce()
            propagated = torch.sparse.mm(operator, matrix)
            return propagated.reshape(n, b, c, t).permute(1, 2, 0, 3).contiguous()
        # A dense tensor is accepted for adapter-level diagnostics.
        return torch.einsum("ncvl,vw->ncwl", (x, support)).contiguous()


def _as_spatial_conv(layer: nn.Conv1d) -> nn.Conv2d:
    if len(layer.kernel_size) != 2:
        raise ValueError("upstream convolution layout changed; inspect pinned source")
    replacement = nn.Conv2d(
        layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size,
        stride=(1, layer.stride[0]), padding=(0, layer.padding[0]),
        dilation=(1, layer.dilation[0]), groups=layer.groups,
        bias=layer.bias is not None, padding_mode=layer.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(layer.weight)
        if layer.bias is not None:
            replacement.bias.copy_(layer.bias)
    return replacement


class ChargedFixedGraphWaveNet(nn.Module):
    """Shared-parameter Graph WaveNet accepting variable CHARGED node counts."""

    def __init__(self, *, channels: int = 2, seq_len: int = 168,
                 pred_len: int = 24, hidden: int = 32, blocks: int = 1,
                 layers: int = 8, dropout: float = 0.3,
                 support_count: int = 2):
        super().__init__()
        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.network = _OFFICIAL.gwnet(
            device=torch.device("cpu"), num_nodes=1, dropout=float(dropout),
            supports=[torch.empty(1, 1) for _ in range(int(support_count))],
            gcn_bool=True, addaptadj=False, aptinit=None, in_dim=self.channels,
            out_dim=self.pred_len, residual_channels=int(hidden),
            dilation_channels=int(hidden), skip_channels=int(hidden) * 8,
            end_channels=int(hidden) * 16, kernel_size=2,
            blocks=int(blocks), layers=int(layers),
        )
        for name in ("gate_convs", "residual_convs", "skip_convs"):
            setattr(self.network, name, nn.ModuleList(
                _as_spatial_conv(layer) for layer in getattr(self.network, name)
            ))
        for layer in self.network.gconv:
            layer.nconv = _FixedNConv()

    def forward(self, x: torch.Tensor,
                supports: Sequence[FixedSupport]) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels or x.shape[3] != self.seq_len:
            raise ValueError(
                f"expected [batch,{self.channels},nodes,{self.seq_len}], got {tuple(x.shape)}"
            )
        if len(supports) != self.network.supports_len:
            raise ValueError(f"expected {self.network.supports_len} supports, got {len(supports)}")
        n = int(x.shape[2])
        prepared = [support.to(x.device) for support in supports]
        if any(support.nodes != n for support in prepared):
            raise ValueError("all graph supports must match x node count")
        # The pinned network reads this ordinary list in each forward pass;
        # no learned node-ID embedding exists in this fixed-graph variant.
        self.network.supports = prepared
        output = self.network(x)
        return output[..., -1]


__all__ = ["ChargedFixedGraphWaveNet", "FixedSupport", "distance_support"]

"""Shared-station N-BEATS-style forecaster with a history-conditioned graph residual.

This is an independent baseline implementation, not a scprc of any
named paper. The graph descriptor and bounded, aggregate-preserving residual
are adapted from the local CHARGED history-behavior-graph experiment; the
forecast backbone is replaced with shared generic N-BEATS-style blocks.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT = 168
HORIZON = 24
BEHAVIOR_DESCRIPTOR_DIM = 26

EXPERIMENT_METADATA: dict[str, Any] = {
    "model_id": "charged-shared-station-nbeats-behavior-graph-residual-v1",
    "method_scope": "independent N-BEATS-style baseline implementation",
    "original_paper_reproduction": False,
    "reproduction_claim": "not a reproduction of any named paper",
    "protocol_adaptation": "SCPRC frozen CHARGED LOCO protocol; runner integration is separate",
    "input": "source-scaled historical volume only, [batch,168,sites]",
    "output": "source-scaled forecast, [batch,24,sites]",
    "scaler": "must be fit from source cities only by the experiment runner",
    "evaluation_matrix": {
        "dataset": "CHARGED v3",
        "targets": ["AMS", "JHB", "LOA", "MEL", "SPO", "SZH"],
        "seeds": [13, 17, 29],
        "training_window": "source-only April-July prefix",
        "configured_source_minibatches": 2048,
        "windows_per_source_minibatch": 4,
        "configured_source_windows": 8192,
        "gradient_aggregation": "PCGrad-compatible; supplied by runner, not embedded",
        "august_role": "retrospective development/audit only, not pristine confirmation",
    },
    "station_identity_or_geometry_used": False,
    "target_future_labels_used_by_model": False,
}


class GenericNBEATSBlock(nn.Module):
    """A generic fully connected N-BEATS-style backcast/forecast block."""

    def __init__(self, context: int, horizon: int, hidden: int,
                 *, include_backcast: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(context, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, context) if include_backcast else None
        self.forecast = nn.Linear(hidden, horizon)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        features = self.trunk(residual)
        backcast = self.backcast(features) if self.backcast is not None else None
        return backcast, self.forecast(features)


class SharedStationNBEATS(nn.Module):
    """Generic N-BEATS-style stacks with one shared set of weights per station."""

    def __init__(self, *, context: int = CONTEXT, horizon: int = HORIZON,
                 hidden: int = 128, blocks: int = 4):
        super().__init__()
        if context != CONTEXT or horizon != HORIZON:
            raise ValueError(f"this CHARGED baseline requires {CONTEXT}->{HORIZON}")
        if hidden <= 0 or blocks <= 0:
            raise ValueError("hidden and blocks must be positive")
        self.context = int(context)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.blocks = nn.ModuleList(
            GenericNBEATSBlock(
                self.context, self.horizon, self.hidden,
                include_backcast=index < int(blocks) - 1,
            )
            for index in range(int(blocks))
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != self.context:
            raise ValueError(
                f"expected [batch,{self.context},sites], got {tuple(history.shape)}"
            )
        batch, _, sites = history.shape
        residual = history.transpose(1, 2).reshape(batch * sites, self.context)
        forecast = residual.new_zeros((batch * sites, self.horizon))
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            if backcast is not None:
                residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast.reshape(batch, sites, self.horizon).transpose(1, 2).contiguous()


class SharedStationNBEATSBehaviorGraphResidual(nn.Module):
    """Shared-site N-BEATS forecast plus a dynamic behavior-graph correction.

    The graph is recomputed from each observed 168-hour volume history. Its
    descriptor and query/key projections are shared across nodes and cities.
    A bounded peer correction is centered across sites, preserving each
    forecast origin's city aggregate. A one-site city receives the backbone
    forecast unchanged.
    """

    def __init__(self, *, hidden: int = 128, blocks: int = 4,
                 graph_rank: int = 16, temperature: float = 0.15,
                 residual_bound: float = 0.20):
        super().__init__()
        if graph_rank <= 0:
            raise ValueError("graph_rank must be positive")
        if temperature <= 0 or residual_bound <= 0:
            raise ValueError("temperature and residual_bound must be positive")
        self.backbone = SharedStationNBEATS(hidden=hidden, blocks=blocks)
        self.graph_rank = int(graph_rank)
        self.temperature = float(temperature)
        self.residual_bound = float(residual_bound)
        self.graph_query = nn.Linear(BEHAVIOR_DESCRIPTOR_DIM, self.graph_rank, bias=False)
        self.graph_key = nn.Linear(BEHAVIOR_DESCRIPTOR_DIM, self.graph_rank, bias=False)
        self.raw_gain = nn.Parameter(torch.tensor(0.54930615))

    @staticmethod
    def behavior_descriptor(history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != CONTEXT:
            raise ValueError(f"expected [batch,{CONTEXT},sites], got {tuple(history.shape)}")
        batch, _, sites = history.shape
        weekly = history.detach().reshape(batch, 7, 24, sites)
        daily_profile = weekly.mean(dim=1).transpose(1, 2)
        local_mean = weekly.mean(dim=(1, 2)).unsqueeze(-1)
        local_std = weekly.std(dim=(1, 2), unbiased=False).unsqueeze(-1)
        profile_std = daily_profile.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        profile_shape = (
            daily_profile - daily_profile.mean(dim=-1, keepdim=True)
        ) / profile_std
        descriptor = torch.cat(
            (profile_shape, local_mean / 5.0, torch.log1p(local_std)), dim=-1
        )
        return F.normalize(descriptor, p=2, dim=-1, eps=1e-6)

    def dynamic_graph(self, history: torch.Tensor) -> torch.Tensor:
        batch, _, sites = history.shape
        if sites < 2:
            return history.new_zeros((batch, sites, sites))
        descriptor = self.behavior_descriptor(history)
        query = F.normalize(self.graph_query(descriptor), dim=-1, eps=1e-6)
        key = F.normalize(self.graph_key(descriptor), dim=-1, eps=1e-6)
        similarity = torch.bmm(query, key.transpose(1, 2))
        diagonal = torch.eye(sites, device=history.device, dtype=torch.bool).unsqueeze(0)
        similarity = similarity.masked_fill(diagonal, torch.finfo(similarity.dtype).min)
        return torch.softmax(
            similarity * (self.graph_rank ** -0.5) / self.temperature, dim=-1
        )

    def forward(self, history: torch.Tensor, *, return_diagnostics: bool = False):
        if history.ndim != 3 or history.shape[1] != CONTEXT:
            raise ValueError(f"expected [batch,{CONTEXT},sites], got {tuple(history.shape)}")
        base = self.backbone(history)
        sites = history.shape[-1]
        adjacency = self.dynamic_graph(history)
        if sites < 2:
            if return_diagnostics:
                return base, {"adjacency": adjacency, "residual": torch.zeros_like(base)}
            return base
        neighbor_forecast = torch.bmm(adjacency, base.transpose(1, 2)).transpose(1, 2)
        gain = 0.20 * torch.tanh(self.raw_gain)
        raw_delta = self.residual_bound * torch.tanh(
            gain * (neighbor_forecast - base) / self.residual_bound
        )
        delta = 0.5 * (
            raw_delta - raw_delta.mean(dim=-1, keepdim=True)
        )
        prediction = base + delta
        if return_diagnostics:
            return prediction, {"adjacency": adjacency, "residual": delta}
        return prediction


def smoke() -> dict[str, Any]:
    torch.manual_seed(2026)
    model = SharedStationNBEATSBehaviorGraphResidual(hidden=32, blocks=2).eval()
    history = torch.randn(2, CONTEXT, 7, requires_grad=True)
    prediction, diagnostics = model(history, return_diagnostics=True)
    permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
    permuted = model(history[:, :, permutation])
    if prediction.shape != (2, HORIZON, 7) or not torch.isfinite(prediction).all():
        raise AssertionError("invalid forecast shape or values")
    if not torch.allclose(permuted, prediction[:, :, permutation], atol=2e-5, rtol=2e-5):
        raise AssertionError("model is not permutation equivariant")
    base = model.backbone(history)
    if not torch.allclose(prediction.sum(-1), base.sum(-1), atol=2e-5, rtol=2e-5):
        raise AssertionError("graph residual changed the city aggregate")
    if diagnostics["residual"].abs().max() > model.residual_bound + 1e-6:
        raise AssertionError("centered graph residual exceeded its configured bound")
    prediction.square().mean().backward()
    if history.grad is None or not torch.isfinite(history.grad).all():
        raise AssertionError("invalid input gradient")
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("missing or non-finite model gradient")
    return {
        "status": "passed",
        "input_shape": list(history.shape),
        "output_shape": list(prediction.shape),
        "variable_site_output_shape": list(model(torch.randn(1, CONTEXT, 3)).shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "permutation_equivariant": True,
        "city_aggregate_preserved": True,
        "graph_residual_bound": model.residual_bound,
        "original_paper_reproduction": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(smoke(), sort_keys=True))

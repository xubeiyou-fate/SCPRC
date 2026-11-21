"""Source-certified, non-parametric phase-relation residual calibration.

This module addresses city-level negative transfer in the September relation
branch.  A fixed peer-median phase descriptor is used as a sidecar to a frozen
PhaseFormer.  A source-only robust envelope supplies a deterministic trust
region: when the current history is farther from the source-supported relation
than the envelope, the fixed residual budget is projected down.  The envelope
is a safety certificate, not a learned gate and it is never fitted on target
labels.
"""
from __future__ import annotations

import torch
from torch import nn


class SourceCertifiedPhaseRelationCalibrator(nn.Module):
    """Frozen-backbone calibrator with fixed peer-median relation descriptors."""

    def __init__(self, horizon: int = 24, period: int = 24, hidden: int = 16,
                 residual_bound: float = 0.20, trust_threshold: float = 1.0,
                 support_scale: float = 1.0):
        super().__init__()
        self.horizon = int(horizon)
        self.period = int(period)
        self.residual_bound = float(residual_bound)
        self.trust_threshold = float(trust_threshold)
        # A deterministic, source-support projection computed from the target
        # historical prefix.  It is either one or zero in the release policy;
        # it is not a trainable gate or a tuned per-window weight.
        self.support_scale = float(support_scale)
        if self.horizon != self.period or self.horizon != 24:
            raise ValueError("this implementation expects a 24-hour horizon")
        # 24 recent + 24 weekly profile + 24 base forecast + 4 statistics,
        # 24 relation mean + 24 relation magnitude + 1 global disagreement.
        input_dim = 24 + 24 + 24 + 4 + 24 + 24 + 1
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _slope(history: torch.Tensor) -> torch.Tensor:
        recent = history[:, -24:]
        t = torch.linspace(-1.0, 1.0, recent.shape[1], device=history.device,
                           dtype=history.dtype).view(1, -1, 1)
        centered = recent - recent.mean(dim=1, keepdim=True)
        return (centered * t).sum(dim=1) / (t.square().sum() + 1e-6)

    def _relation_features(self, history: torch.Tensor):
        local_mean = history.mean(dim=1, keepdim=True)
        local_std = history.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        normalized = (history - local_mean) / local_std
        phase = normalized.permute(0, 2, 1).reshape(
            history.shape[0], history.shape[2], 7, 24
        ).permute(0, 1, 3, 2).contiguous()
        # Non-parametric peer consensus; no station IDs, coordinates or source
        # labels.  Median is robust to sparse/large-demand station outliers.
        peer = phase.median(dim=1, keepdim=True).values
        delta = peer - phase
        phase_mean = delta.mean(dim=-1)
        phase_magnitude = delta.abs().mean(dim=-1)
        global_disagreement = torch.sqrt(
            delta.square().mean(dim=(1, 2, 3)).clamp_min(1e-8)
        ).view(history.shape[0], 1, 1)
        return phase_mean, phase_magnitude, global_disagreement

    def forward(self, history: torch.Tensor, backbone_forecast: torch.Tensor,
                *, return_residual: bool = False):
        if history.ndim != 3 or backbone_forecast.ndim != 3:
            raise ValueError("history and backbone forecast must be rank-3 tensors")
        if history.shape[0] != backbone_forecast.shape[0] or history.shape[2] != backbone_forecast.shape[2]:
            raise ValueError("history and backbone forecast batch/station axes must match")
        if history.shape[1] != 168 or backbone_forecast.shape[1] != self.horizon:
            raise ValueError("expected 168-hour history and 24-hour forecast")

        recent = history[:, -24:].permute(0, 2, 1)
        profile = history[:, -7 * 24:].reshape(
            history.shape[0], 7, 24, history.shape[2]
        ).mean(dim=1).permute(0, 2, 1)
        base = backbone_forecast.permute(0, 2, 1)
        mean = history.mean(dim=1)
        std = history.std(dim=1, unbiased=False).clamp_min(1e-4)
        last = history[:, -1]
        slope = self._slope(history)
        stats = torch.stack((mean, std, last, slope), dim=-1)
        relation_mean, relation_magnitude, disagreement = self._relation_features(history)
        disagreement_feature = torch.log1p(disagreement).expand(-1, history.shape[2], -1)
        features = torch.cat((recent, profile, base, stats,
                              relation_mean, relation_magnitude,
                              disagreement_feature), dim=-1)
        raw = self.head(features)

        # Fixed source-certified trust-region projection.  The threshold is
        # computed once from source training windows at the runner; no target
        # labels or trainable scalar gate enter this operation.
        trust = (self.trust_threshold / disagreement.clamp_min(1e-6)).clamp(max=1.0)
        local_scale = std.clamp(0.02, 1.0)
        residual = (self.support_scale * self.residual_bound
                    * local_scale.unsqueeze(-1) * trust * torch.tanh(raw))
        residual = residual.permute(0, 2, 1).contiguous()
        corrected = backbone_forecast + residual
        if return_residual:
            return corrected, residual
        return corrected


def source_relation_disagreement(history: torch.Tensor) -> torch.Tensor:
    """Return fixed peer-median disagreement for threshold estimation."""
    if history.ndim != 3 or history.shape[1] != 168:
        raise ValueError("expected [batch,168,stations]")
    mean = history.mean(dim=1, keepdim=True)
    std = history.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    normalized = (history - mean) / std
    phase = normalized.permute(0, 2, 1).reshape(
        history.shape[0], history.shape[2], 7, 24
    ).permute(0, 1, 3, 2).contiguous()
    peer = phase.median(dim=1, keepdim=True).values
    return torch.sqrt((phase - peer).square().mean(dim=(1, 2, 3)).clamp_min(1e-8))


__all__ = ["SourceCertifiedPhaseRelationCalibrator", "source_relation_disagreement"]

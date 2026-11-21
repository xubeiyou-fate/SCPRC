"""Source-certified internal phase adapter for PhaseFormer.

The relation adapter is inserted before PhaseFormer's phase embedding and is
trained end-to-end on source cities.  Its correction is still constrained by
fixed, source-only reliability and source-support projections.  No learned
gate parameter or expert pool is used.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scprc.training.relation_attenuation import (
    ReliabilityCappedPhaseRelationPhaseFormer,
)


def _fixed_peer_median_disagreement(history: torch.Tensor) -> torch.Tensor:
    """Match the source-only q90 statistic used by the LOCO runner."""
    mean = history.mean(dim=1, keepdim=True)
    std = history.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    normalized = (history - mean) / std
    phase = normalized.permute(0, 2, 1).reshape(
        history.shape[0], history.shape[2], 7, 24
    ).permute(0, 1, 3, 2).contiguous()
    peer = phase.median(dim=1, keepdim=True).values
    return torch.sqrt((phase - peer).square().mean(dim=(1, 2, 3)).clamp_min(1e-8))


class SCPRC(
    ReliabilityCappedPhaseRelationPhaseFormer
):
    """End-to-end internal relation adapter with fixed source certificates."""

    def __init__(self, *, support_scale: float = 1.0,
                 source_relation_threshold: float = 1.0,
                 use_scale_gate: bool = True,
                 use_peer_gate: bool = True,
                 use_source_disagreement_gate: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.support_scale = float(support_scale)
        self.source_relation_threshold = float(source_relation_threshold)
        self.use_scale_gate = bool(use_scale_gate)
        self.use_peer_gate = bool(use_peer_gate)
        self.use_source_disagreement_gate = bool(use_source_disagreement_gate)
        if self.support_scale not in (0.0, 1.0):
            raise ValueError("support_scale must be exactly 0 or 1")
        if self.source_relation_threshold <= 0.0:
            raise ValueError("source_relation_threshold must be positive")

    def relation_trust_gate(self, diagnostics: dict[str, torch.Tensor],
                            station_scale: torch.Tensor) -> torch.Tensor:
        """Combine fixed source q90 trust with the fixed peer reliability cap."""
        del station_scale
        disagreement = diagnostics["relation_disagreement"]
        # The first term is the existing input-only peer reliability cap.  The
        # second term is a source-only q90 envelope estimated before training.
        if self.use_peer_gate:
            peer_reference = disagreement.median(dim=1, keepdim=True).values
            peer_trust = (peer_reference / disagreement.clamp_min(1e-6)).clamp(max=1.0)
        else:
            peer_trust = torch.ones_like(disagreement)
        # Apply the source q90 certificate at the whole-window level.  A
        # station-level q90 projection was too aggressive for sparse cities:
        # it suppressed otherwise useful relative corrections and increased
        # seed variance.  The peer term still attenuates atypical stations.
        window_disagreement = diagnostics["source_peer_disagreement"].view(-1, 1)
        if self.use_source_disagreement_gate:
            source_trust = (
                torch.as_tensor(self.source_relation_threshold,
                                device=disagreement.device,
                                dtype=disagreement.dtype)
                / window_disagreement.clamp_min(1e-6)
            ).clamp(max=1.0)
        else:
            source_trust = torch.ones_like(window_disagreement)
        return peer_trust * source_trust

    def forward(self, history: torch.Tensor, *, return_diagnostics: bool = False):
        if history.ndim != 3 or history.shape[1] != 168:
            raise ValueError(f"expected [batch,168,stations], got {tuple(history.shape)}")
        net = self.backbone
        x_norm, stats = net.revin.normalize(history)
        x = x_norm.permute(0, 2, 1).contiguous()
        if net.pad_seq_len > 0:
            x = F.pad(x, (0, net.pad_seq_len), mode="circular")
        x_periods = x.view(x.shape[0], x.shape[1], net.num_periods_input, self.period_len)
        phase_series = net._to_phase_series(x_periods)
        # Match the pinned relation implementation: relation statistics and
        # its zero-initialized correction are formed in FP32 under AMP.
        relation_input = phase_series.float()
        related_series, diagnostics = self.relation(relation_input)
        diagnostics["source_peer_disagreement"] = _fixed_peer_median_disagreement(
            history.float()
        )

        # Keep the source-certified projection in FP32.  This is deterministic
        # and uses only the current history plus a source-derived threshold.
        with torch.autocast(device_type=history.device.type, enabled=False):
            phase_fp32 = relation_input.float()
            related_fp32 = related_series.float()
            relation_delta = related_fp32 - phase_fp32
            station_scale = stats[1].squeeze(1).float()
            scale_gate = (
                self.station_scale_gate(station_scale)
                if self.use_scale_gate else torch.ones_like(station_scale)
            )
            trust_gate = self.relation_trust_gate(diagnostics, station_scale)
            raw_delta = relation_delta * station_scale[:, :, None, None]
            raw_delta = raw_delta * scale_gate[:, :, None, None]
            raw_delta = raw_delta * trust_gate[:, :, None, None]
            raw_delta = self.support_scale * raw_delta
            # Preserve the city-level mean exactly, as in the parent relation
            # adapter, so the correction only transports station phase shape.
            raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            relation_delta = raw_delta / station_scale.clamp_min(1e-6)[:, :, None, None]
            phase_series = phase_fp32 + relation_delta

        latent = net.embedding(phase_series)
        phase_series_cur = phase_series
        for layer_index, unit in enumerate(net.routing_layers):
            latent, phase_steps_in = unit(phase_series_cur, latent)
            if layer_index < len(net.routing_layers) - 1:
                phase_series_cur = phase_steps_in
        phase_steps = net.predictor(latent)
        periods = net._from_phase_steps_to_periods(phase_steps)
        forecast = periods.reshape(periods.shape[0], periods.shape[1], -1)[..., :self.pred_len]
        forecast = forecast.permute(0, 2, 1).contiguous()
        forecast = net.revin.denormalize(forecast, stats)
        if return_diagnostics:
            diagnostics = {
                **diagnostics,
                "source_relation_threshold": torch.as_tensor(
                    self.source_relation_threshold, device=history.device
                ),
                "support_scale": torch.as_tensor(
                    self.support_scale, device=history.device
                ),
            }
            return forecast, diagnostics
        return forecast


__all__ = ["SCPRC"]

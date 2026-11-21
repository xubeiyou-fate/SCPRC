"""Phase-conditioned station relations with a source-free scale cap.

The relation still sees only the PhaseFormer RevIN-normalized 168-hour input.
Its learned correction is reduced at high-volatility stations by the ratio
between the city's current median station scale and that station's scale.
After PhaseFormer denormalization, this caps the relation's raw-unit change
relative to the city's typical station, without reading IDs or future labels.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scprc.training.phase_relations import (
    InputPhaseRelationPhaseFormer,
    RelationMode,
)


class ScaleCappedPhaseRelationPhaseFormer(InputPhaseRelationPhaseFormer):
    """PhaseFormer relation whose raw correction is capped by local scale."""

    @staticmethod
    def station_scale_gate(scale: torch.Tensor) -> torch.Tensor:
        """Return min(1, city-median-scale / station-scale), per sample/site."""
        if scale.ndim != 2 or scale.shape[1] < 1:
            raise ValueError("scale must have shape [batch, nonempty station set]")
        reference = scale.median(dim=1, keepdim=True).values
        return (reference / scale.clamp_min(1e-6)).clamp(max=1.0)

    @staticmethod
    def relation_trust_gate(diagnostics: dict[str, torch.Tensor],
                            station_scale: torch.Tensor) -> torch.Tensor:
        """Identity gate for the scale-only arm; overridden by the trust arm."""
        del diagnostics
        return torch.ones_like(station_scale)

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
        relation_input = phase_series.float()
        related_series, diagnostics = self.relation(relation_input)

        # RevIN scale is computed only from this origin's observed history. Its
        # ratio is invariant to source-city global scaling and target city size.
        with torch.autocast(device_type=history.device.type, enabled=False):
            phase_series = relation_input
            relation_delta = related_series.float() - phase_series
            station_scale = stats[1].squeeze(1).float()
            scale_gate = self.station_scale_gate(station_scale)
            trust_gate = self.relation_trust_gate(diagnostics, station_scale)
            raw_delta = relation_delta * station_scale[:, :, None, None]
            raw_delta = raw_delta * scale_gate[:, :, None, None]
            raw_delta = raw_delta * trust_gate[:, :, None, None]
            raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            relation_delta = raw_delta / station_scale.clamp_min(1e-6)[:, :, None, None]
            phase_series = phase_series + relation_delta

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
                "scale_gate_mean": scale_gate.detach().mean(),
                "scale_gate_min": scale_gate.detach().min(),
                "scale_gate_max": scale_gate.detach().max(),
                "relation_trust_mean": trust_gate.detach().mean(),
                "station_scale_median": station_scale.detach().median(),
            }
            return forecast, diagnostics
        return forecast


def build_scale_capped_model(*, channels: int, relation_mode: RelationMode,
                             latent_dim: int = 32, phase_layers: int = 2,
                             routers: int = 1, heads: int = 8,
                             dropout: float = 0.1, pos_embed: bool = True,
                             relation_hidden: int = 16, relation_rank: int = 8):
    return ScaleCappedPhaseRelationPhaseFormer(
        channels=channels, seq_len=168, pred_len=24, period_len=24,
        latent_dim=latent_dim, phase_layers=phase_layers, routers=routers,
        heads=heads, dropout=dropout, pos_embed=pos_embed,
        relation_mode=relation_mode, relation_hidden=relation_hidden,
        relation_rank=relation_rank,
    )


class ReliabilityCappedPhaseRelationPhaseFormer(ScaleCappedPhaseRelationPhaseFormer):
    """Scale-capped relation, additionally attenuated for atypical histories."""

    @staticmethod
    def relation_trust_gate(diagnostics: dict[str, torch.Tensor],
                            station_scale: torch.Tensor) -> torch.Tensor:
        del station_scale
        disagreement = diagnostics["relation_disagreement"]
        reference = disagreement.median(dim=1, keepdim=True).values
        return (reference / disagreement.clamp_min(1e-6)).clamp(max=1.0)


def build_reliability_capped_model(*, channels: int, relation_mode: RelationMode,
                                   latent_dim: int = 32, phase_layers: int = 2,
                                   routers: int = 1, heads: int = 8,
                                   dropout: float = 0.1, pos_embed: bool = True,
                                   relation_hidden: int = 16, relation_rank: int = 8):
    return ReliabilityCappedPhaseRelationPhaseFormer(
        channels=channels, seq_len=168, pred_len=24, period_len=24,
        latent_dim=latent_dim, phase_layers=phase_layers, routers=routers,
        heads=heads, dropout=dropout, pos_embed=pos_embed,
        relation_mode=relation_mode, relation_hidden=relation_hidden,
        relation_rank=relation_rank,
    )

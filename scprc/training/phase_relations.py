"""Historical station relations before PhaseFormer phase tokenization.

The operator sees only PhaseFormer's RevIN-normalized 168-hour history.  It
shares weights over an unordered, variable-size station set and mixes each
hour-of-day phase independently before the pinned PhaseFormer embedding and
router stack.  No coordinates, station IDs, adjacency, or future values are
used.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
_PHASEFORMER_ADAPTERS = _ROOT / "scprc" / "baselines"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_PHASEFORMER_ADAPTERS) not in sys.path:
    sys.path.insert(0, str(_PHASEFORMER_ADAPTERS))
from scprc.baselines.phaseformer_adapter import PhaseFormerUrbanEV


RelationMode = Literal["phase", "phase_agnostic_daily", "uniform"]


class HistoricalPhaseRelation(nn.Module):
    """Positive-kernel station mixing on [B,N,24,7] normalized phase inputs."""

    def __init__(self, cycles: int = 7, hidden: int = 16, rank: int = 8,
                 mode: RelationMode = "phase"):
        super().__init__()
        if mode not in {"phase", "phase_agnostic_daily", "uniform"}:
            raise ValueError(f"unknown relation mode {mode!r}")
        self.cycles = int(cycles)
        self.mode = mode
        self.profile = nn.Sequential(nn.Linear(cycles, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.query = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(hidden, rank, bias=False)
        self.correction = nn.Sequential(nn.Linear(cycles, hidden), nn.GELU(),
                                        nn.Linear(hidden, cycles))
        # Exact PhaseFormer identity at initialization makes paired training
        # start from the strong backbone rather than a random perturbation.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, phase_series: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if phase_series.ndim != 4:
            raise ValueError("phase_series must be [batch, stations, 24, cycles]")
        batch, nodes, phases, cycles = phase_series.shape
        if cycles != self.cycles or nodes < 1:
            raise ValueError(f"expected a nonempty station set and {self.cycles} cycles")

        # Keep learned kernel scores and O(N) reductions in FP32 even when the
        # backbone is under autocast. For large cities, FP16 station sums can
        # overflow and produce inf/inf in the normalized relation.
        output_dtype = phase_series.dtype
        device_type = phase_series.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            phase_fp32 = phase_series.float()
            values = phase_fp32 - phase_fp32.mean(dim=1, keepdim=True)
            if self.mode == "uniform":
                q = phase_fp32.new_ones(batch, nodes, phases, 1)
                k = phase_fp32.new_ones(batch, nodes, phases, 1)
            elif self.mode == "phase":
                encoded = self.profile(phase_fp32)
                q = F.elu(self.query(encoded)) + 1.0
                k = F.elu(self.key(encoded)) + 1.0
            else:
                # A single station relation per window, conditioned on its
                # seven daily mean levels; shared across the 24 phases.
                daily_level = phase_fp32.mean(dim=2)
                encoded = self.profile(daily_level)
                q = (F.elu(self.query(encoded)) + 1.0).unsqueeze(2).expand(-1, -1, phases, -1)
                k = (F.elu(self.key(encoded)) + 1.0).unsqueeze(2).expand(-1, -1, phases, -1)

            # R_i = q_i^T sum_j(k_j v_j) / q_i^T sum_j(k_j), without N×N.
            kv = torch.einsum("bnlr,bnlp->blrp", k, values)
            key_mass = k.sum(dim=1)
            numerator = torch.einsum("bnlr,blrp->bnlp", q, kv)
            denominator = torch.einsum("bnlr,blr->bnl", q, key_mass).clamp_min(1e-6)
            related = numerator / denominator.unsqueeze(-1)
            relation_disagreement = (phase_fp32 - related).square().mean(dim=(-1, -2)).add(1e-8).sqrt()
            correction = self.correction(related)
            correction = correction - correction.mean(dim=1, keepdim=True)
            mixed = phase_fp32 + correction
        diagnostics = {
            "relation_abs_mean": related.detach().abs().mean(),
            "correction_abs_mean": correction.detach().abs().mean(),
            "denominator_min": denominator.detach().min(),
            "query_phase_variation": (q - q[:, :, :1]).detach().abs().mean(),
            # Kept attached to the graph for input-only reliability gating.
            "relation_disagreement": relation_disagreement,
        }
        return mixed.to(dtype=output_dtype), diagnostics


class InputPhaseRelationPhaseFormer(nn.Module):
    """Pinned PhaseFormer with input-phase station relation correction."""

    def __init__(self, *, channels: int, seq_len: int = 168, pred_len: int = 24,
                 period_len: int = 24, latent_dim: int = 32,
                 phase_layers: int = 2, routers: int = 1, heads: int = 8,
                 dropout: float = 0.1, pos_embed: bool = True,
                 relation_mode: RelationMode = "phase", relation_hidden: int = 16,
                 relation_rank: int = 8):
        super().__init__()
        self.backbone = PhaseFormerUrbanEV(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            period_len=period_len, latent_dim=latent_dim,
            phase_layers=phase_layers, routers=routers, heads=heads,
            dropout=dropout, pos_embed=pos_embed,
        ).network
        if not self.backbone.use_revin:
            raise ValueError("input-phase relation requires the pinned RevIN normalization")
        if self.backbone.num_periods_input != 7 or period_len != 24:
            raise ValueError("this experiment is fixed to 168 hours as 7×24 phases")
        self.relation = HistoricalPhaseRelation(
            cycles=self.backbone.num_periods_input, hidden=relation_hidden,
            rank=relation_rank, mode=relation_mode,
        )
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)

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
        phase_series, diagnostics = self.relation(phase_series)

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
            return forecast, diagnostics
        return forecast


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


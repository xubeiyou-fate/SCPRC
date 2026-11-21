"""Phase-wise DeepSets relation adapter and PhaseFormer wrapper.

The adapter replaces only the PhaseFormer input relation operator. Each daily
phase is processed independently; station elements share an encoder and a
decoder, while mean pooling makes the relation permutation invariant.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from scprc.baselines.phaseformer_adapter import PhaseFormerUrbanEV


class PhaseWiseDeepSetsRelationAdapter(nn.Module):
    """DeepSets adapter for ``[batch, stations, 24, 7]`` phase profiles."""

    def __init__(self, cycles: int = 7, phases: int = 24,
                 encoder_hidden: int = 31, embedding_dim: int = 8,
                 decoder_hidden: int = 16, residual_bound: float = 0.2):
        super().__init__()
        if cycles != 7 or phases != 24:
            raise ValueError("expected 24 phases with seven daily cycles")
        if (encoder_hidden, embedding_dim, decoder_hidden) != (31, 8, 16):
            raise ValueError("adapter widths are fixed by the frozen parameter match")
        if residual_bound <= 0:
            raise ValueError("residual_bound must be positive")

        self.cycles = cycles
        self.phases = phases
        self.embedding_dim = embedding_dim
        self.residual_bound = float(residual_bound)
        self.encoder = nn.Sequential(
            nn.Linear(cycles, encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, cycles),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, phase_series: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if phase_series.ndim != 4:
            raise ValueError("phase_series must be [batch, stations, 24, 7]")
        batch, stations, phases, cycles = phase_series.shape
        if stations < 1 or phases != self.phases or cycles != self.cycles:
            raise ValueError("expected a nonempty station set and phase shape [24,7]")

        output_dtype = phase_series.dtype
        with torch.autocast(device_type=phase_series.device.type, enabled=False):
            phase_fp32 = phase_series.float()
            phase_tokens = phase_fp32.permute(0, 2, 1, 3).reshape(
                batch * phases, stations, cycles
            )
            encoded = self.encoder(phase_tokens)
            context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
            raw_correction = self.decoder(torch.cat((encoded, context), dim=-1))
            raw_correction = raw_correction.reshape(
                batch, phases, stations, cycles
            ).permute(0, 2, 1, 3)
            centered = raw_correction - raw_correction.mean(dim=1, keepdim=True)
            bounded = self.residual_bound * torch.tanh(
                centered / self.residual_bound
            )
            bounded = bounded - bounded.mean(dim=1, keepdim=True)
            max_abs = bounded.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            bounded = bounded * (self.residual_bound / max_abs).clamp(max=1.0)
            mixed = phase_fp32 + bounded

        diagnostics = {
            "correction_abs_mean": bounded.detach().abs().mean(),
            "correction_abs_max": bounded.detach().abs().max(),
        }
        return mixed.to(dtype=output_dtype), diagnostics


class PhaseFormerDeepSetsAdapter(nn.Module):
    """Pinned PhaseFormer backbone with only its input relation swapped."""

    def __init__(self, *, channels: int, seq_len: int = 168,
                 pred_len: int = 24, period_len: int = 24,
                 latent_dim: int = 32, phase_layers: int = 2,
                 routers: int = 1, heads: int = 8, dropout: float = 0.1,
                 pos_embed: bool = True, residual_bound: float = 0.2):
        super().__init__()
        if seq_len != 168 or pred_len != 24 or period_len != 24:
            raise ValueError("this model expects 168-hour input and 24-hour output")
        self.backbone = PhaseFormerUrbanEV(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            period_len=period_len,
            latent_dim=latent_dim,
            phase_layers=phase_layers,
            routers=routers,
            heads=heads,
            dropout=dropout,
            pos_embed=pos_embed,
        ).network
        if not self.backbone.use_revin:
            raise ValueError("the relation adapter requires PhaseFormer RevIN")
        if self.backbone.num_periods_input != 7:
            raise ValueError("the adapter requires seven daily cycles")
        self.relation = PhaseWiseDeepSetsRelationAdapter(
            residual_bound=residual_bound
        )
        self.pred_len = int(pred_len)

    def forward(self, history: torch.Tensor, *,
                return_diagnostics: bool = False) -> Any:
        if history.ndim != 3 or history.shape[1] != 168:
            raise ValueError(f"expected [batch,168,stations], got {tuple(history.shape)}")

        net = self.backbone
        x_norm, stats = net.revin.normalize(history)
        x = x_norm.permute(0, 2, 1).contiguous()
        if net.pad_seq_len > 0:
            x = F.pad(x, (0, net.pad_seq_len), mode="circular")
        x_periods = x.view(
            x.shape[0], x.shape[1], net.num_periods_input, net.period_len
        )
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
        forecast = periods.reshape(periods.shape[0], periods.shape[1], -1)[
            ..., :self.pred_len
        ]
        forecast = forecast.permute(0, 2, 1).contiguous()
        forecast = net.revin.denormalize(forecast, stats)
        if return_diagnostics:
            return forecast, diagnostics
        return forecast


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters()
               if parameter.requires_grad)


__all__ = [
    "PhaseFormerDeepSetsAdapter",
    "PhaseWiseDeepSetsRelationAdapter",
    "trainable_parameter_count",
]

"""Parameter-matched phase-wise self-attention adapter for PhaseFormer.

The adapter consumes the same RevIN-normalized phase tensor as
``HistoricalPhaseRelation``: ``[batch, stations, 24, 7]``. For each of the 24
hour-of-day phases it applies dense self-attention over the variable-size,
unordered station set, then emits a station-centered, bounded correction before
PhaseFormer's phase embedding. The default adapter has 897 trainable parameters,
0.66% fewer than the 903-parameter SCPRC relation adapter.

Attention compute is O(batch * 24 * stations^2 * embed_dim), with a dense score
tensor of O(batch * 24 * stations^2) elements in the unfused implementation.
Fused scaled-dot-product-attention kernels may avoid materializing the full
score tensor, but they do not remove the quadratic compute dependence.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from scprc.baselines.phaseformer_adapter import PhaseFormerUrbanEV


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters()
               if parameter.requires_grad)


def dense_attention_score_elements(batch_size: int, stations: int,
                                   phases: int = 24,
                                   heads: int = 1) -> int:
    if min(batch_size, stations, phases, heads) < 1:
        raise ValueError("batch_size, stations, phases, and heads must be positive")
    return batch_size * phases * heads * stations * stations


def dense_attention_score_bytes(batch_size: int, stations: int,
                                phases: int = 24, heads: int = 1,
                                element_size: int = 4) -> int:
    if element_size < 1:
        raise ValueError("element_size must be positive")
    return dense_attention_score_elements(batch_size, stations, phases, heads) * element_size


class PhaseWiseAttentionSetAdapter(nn.Module):
    """Generic dense self-attention over stations, independently per phase.

    Input/output follows ``HistoricalPhaseRelation`` exactly and can replace
    its relation module without changing PhaseFormer integration code.
    """

    def __init__(self, cycles: int = 7, phases: int = 24,
                 embed_dim: int = 8, feedforward_dim: int = 26,
                 residual_bound: float = 0.2):
        super().__init__()
        if cycles != 7 or phases != 24:
            raise ValueError("the PhaseFormer adapter expects 24 phases × 7 cycles")
        if embed_dim != 8 or feedforward_dim != 26:
            raise ValueError("default dimensions are fixed for parameter matching")
        if residual_bound <= 0:
            raise ValueError("residual_bound must be positive")

        self.cycles = cycles
        self.phases = phases
        self.embed_dim = embed_dim
        self.residual_bound = float(residual_bound)
        self.profile = nn.Linear(cycles, embed_dim)
        self.norm_attention = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_feedforward = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.correction = nn.Linear(embed_dim, cycles)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

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
            encoded = self.profile(phase_tokens)
            attention_input = self.norm_attention(encoded)
            attended, _ = self.attention(
                attention_input,
                attention_input,
                attention_input,
                need_weights=False,
            )
            encoded = encoded + attended
            encoded = encoded + self.feedforward(self.norm_feedforward(encoded))

            raw_correction = self.correction(encoded).reshape(
                batch, phases, stations, cycles
            ).permute(0, 2, 1, 3)
            centered = raw_correction - raw_correction.mean(dim=1, keepdim=True)
            bounded = self.residual_bound * torch.tanh(centered / self.residual_bound)
            bounded = bounded - bounded.mean(dim=1, keepdim=True)
            max_abs = bounded.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            bounded = bounded * (self.residual_bound / max_abs).clamp(max=1.0)
            mixed = phase_fp32 + bounded

        diagnostics = {
            "correction_abs_mean": bounded.detach().abs().mean(),
            "correction_abs_max": bounded.detach().abs().max(),
            "attention_score_elements": torch.tensor(
                dense_attention_score_elements(batch, stations, phases),
                device=phase_series.device,
            ),
        }
        return mixed.to(dtype=output_dtype), diagnostics


class PhaseWiseAttentionPhaseFormer(nn.Module):
    """PhaseFormer exact backbone with the attention adapter at SCPRC's input.

    ``history`` is the source-scaled ``[batch, 168, stations]`` tensor used by
    the existing PhaseFormer LOCO runners. The backbone's own RevIN produces
    the normalized ``[batch, stations, 24, 7]`` adapter input. The public
    ``relation`` attribute has the same call contract as
    ``HistoricalPhaseRelation``.
    """

    def __init__(self, *, channels: int, seq_len: int = 168,
                 pred_len: int = 24, period_len: int = 24,
                 latent_dim: int = 32, phase_layers: int = 2,
                 routers: int = 1, heads: int = 8, dropout: float = 0.1,
                 pos_embed: bool = True, residual_bound: float = 0.2):
        super().__init__()
        if seq_len != 168 or pred_len != 24 or period_len != 24:
            raise ValueError("this baseline expects 168-hour input and 24-hour output")
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
            raise ValueError("the attention adapter requires PhaseFormer RevIN")
        if self.backbone.num_periods_input != 7:
            raise ValueError("the attention adapter requires seven daily cycles")
        self.relation = PhaseWiseAttentionSetAdapter(
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
            x = torch.nn.functional.pad(x, (0, net.pad_seq_len), mode="circular")
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


__all__ = [
    "PhaseWiseAttentionPhaseFormer",
    "PhaseWiseAttentionSetAdapter",
    "dense_attention_score_bytes",
    "dense_attention_score_elements",
    "trainable_parameter_count",
]

"""Canonical SCPRC design controls; no data loading or training at import.

Every variant starts with the canonical Full constructor inside an isolated
CPU RNG context. Only the requested branch is then replaced. In particular,
the strict shared operator averages *normalized* phase-specific relations,
and the local control retains deterministic cross-station statistics.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'scprc/training').is_dir()
)
EXPERIMENTS = ROOT / 'scprc/training'
for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Same compatibility shim as the canonical anchor runner: this legacy module
# is imported transitively but never used by canonical PCGrad construction.
_legacy_name = "calibration_protocol_constants"
if _legacy_name not in sys.modules:
    _legacy = types.ModuleType(_legacy_name)
    _legacy.CONTEXT, _legacy.HORIZON, _legacy.TRAIN_END = 168, 24, 2928
    sys.modules[_legacy_name] = _legacy

import scprc.training.train_scprc as pcgrad  # noqa: E402
from scprc.models.scprc import (  # noqa: E402
    SCPRC,
    _fixed_peer_median_disagreement,
)


VARIANTS = (
    "full_same_run", "phase_shared_strict",
    "stationwise_matched", "concat_project",
)
VARIANTS4 = VARIANTS


class PhaseSharedStrictRelation(nn.Module):
    """Reuse Full's exact parameters, averaging A_s after normalization.

    The canonical relation instance is promoted to this class without calling
    another constructor. Q/K retain their [B,N,source_phase,rank] profiles.
    The temporary K_s^T V_p has size B*P*P*rank*cycles, independent of N;
    neither forward nor backward requires any station-by-station matrix.
    """

    @staticmethod
    def aggregate(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor):
        """Return mean_s(A_s) V_p and the original source-phase masses.

        q/k: [B,N,S,rank], values: [B,N,P,cycles]. Kept separate for
        small-N Q/K/V autograd comparisons with an explicit dense oracle.
        """
        denominator = torch.einsum("bnsr,bsr->bns", q, k.sum(dim=1)).clamp_min(1e-6)
        kv = torch.einsum("bnsr,bnpd->bsrpd", k, values)
        related = torch.einsum(
            "bnsr,bsrpd->bnpd", q / denominator.unsqueeze(-1), kv / q.shape[2]
        )
        return related, denominator

    def forward(self, phase_series: torch.Tensor):
        if phase_series.ndim != 4:
            raise ValueError("phase_series must be [batch, stations, 24, cycles]")
        _, nodes, phases, cycles = phase_series.shape
        if cycles != self.cycles or nodes < 1 or phases < 1:
            raise ValueError(f"expected a nonempty station/phase set and {self.cycles} cycles")
        with torch.autocast(device_type=phase_series.device.type, enabled=False):
            z = phase_series.float()
            values = z - z.mean(dim=1, keepdim=True)
            encoded = self.profile(z)
            q = F.elu(self.query(encoded)) + 1.0
            k = F.elu(self.key(encoded)) + 1.0
            # Normalize each source phase BEFORE averaging operators. Dividing
            # a sum of unnormalized phase kernels by a pooled mass is different.
            related, denominator = self.aggregate(q, k, values)
            disagreement = (z - related).square().mean(dim=(-1, -2)).add(1e-8).sqrt()
            correction = self.correction(related)
            correction = correction - correction.mean(dim=1, keepdim=True)
            mixed = z + correction
        return mixed.to(dtype=phase_series.dtype), {
            "relation_abs_mean": related.detach().abs().mean(),
            "correction_abs_mean": correction.detach().abs().mean(),
            "denominator_min": denominator.detach().min(),
            "query_phase_variation": (q - q[:, :, :1]).detach().abs().mean(),
            "relation_disagreement": disagreement,
            "operator_shared_across_phases": z.new_tensor(True, dtype=torch.bool),
        }


class StationwiseMatchedRelation(nn.Module):
    """907 registered trainable parameters; no learned cross-station mixing.

    Common station centering and outer scale/source statistics remain. Since
    R=V=Z-mean_stations(Z), RMS(Z-R) is identical at every station: the peer
    gate is mathematically one. Compute that residual from the common mean
    directly to avoid roundoff from subtracting Z-(Z-mean(Z)).
    The final seven biases cancel under station centering, as in Full; the
    registered trainable count does not claim nonzero gradients for them.
    """

    def __init__(self, cycles: int = 7):
        super().__init__()
        self.cycles = cycles
        self.mode = "stationwise_matched"
        self.correction = nn.Sequential(
            nn.Linear(cycles, 60), nn.GELU(), nn.Linear(60, cycles),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, phase_series: torch.Tensor):
        if phase_series.ndim != 4:
            raise ValueError("phase_series must be [batch, stations, 24, cycles]")
        batch, nodes, phases, cycles = phase_series.shape
        if cycles != self.cycles or nodes < 1 or phases < 1:
            raise ValueError(f"expected a nonempty station/phase set and {self.cycles} cycles")
        with torch.autocast(device_type=phase_series.device.type, enabled=False):
            z = phase_series.float()
            common_mean = z.mean(dim=1, keepdim=True)
            related = z - common_mean
            disagreement = common_mean.square().mean(dim=(-1, -2)).add(1e-8).sqrt()
            # Materialize the broadcast for the canonical median reduction;
            # some CPU median kernels mishandle a zero-stride station axis.
            disagreement = disagreement.expand(batch, nodes).contiguous()
            correction = self.correction(related)
            correction = correction - correction.mean(dim=1, keepdim=True)
            mixed = z + correction
        return mixed.to(dtype=phase_series.dtype), {
            "relation_abs_mean": related.detach().abs().mean(),
            "correction_abs_mean": correction.detach().abs().mean(),
            # Identity aggregation has unit denominator and no Q/K profile.
            "denominator_min": z.new_ones(()),
            "query_phase_variation": z.new_zeros(()),
            "relation_disagreement": disagreement,
            "peer_gate_degenerate": z.new_tensor(True, dtype=torch.bool),
        }


class _DiagnosticControlAdapter(SCPRC):
    """Canonical gating order, exposing diagnostics for the new controls.

    Promoted from a fully constructed canonical model by build_model. The
    forward follows canonical Full verbatim through the second centering and
    inverse scaling. The default fusion is canonical addition. Full itself
    retains the original class and forward, without this diagnostics wrapper.
    """

    def _fuse(self, phase: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return phase + delta

    def _fusion_diagnostics(self, phase, fused):
        return {}

    def forward(self, history: torch.Tensor, *, return_diagnostics: bool = False):
        if history.ndim != 3 or history.shape[1] != 168:
            raise ValueError(f"expected [batch,168,stations], got {tuple(history.shape)}")
        net = self.backbone
        x_norm, stats = net.revin.normalize(history)
        x = x_norm.permute(0, 2, 1).contiguous()
        if net.pad_seq_len > 0:
            x = F.pad(x, (0, net.pad_seq_len), mode="circular")
        x_periods = x.view(x.shape[0], x.shape[1], net.num_periods_input, self.period_len)
        relation_input = net._to_phase_series(x_periods).float()
        related_series, diagnostics = self.relation(relation_input)
        diagnostics["source_peer_disagreement"] = _fixed_peer_median_disagreement(history.float())

        with torch.autocast(device_type=history.device.type, enabled=False):
            phase_fp32 = relation_input.float()
            relation_delta = related_series.float() - phase_fp32
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
            if return_diagnostics:
                gated_before = raw_delta.detach().abs().mean()
            raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
            relation_delta = raw_delta / station_scale.clamp_min(1e-6)[:, :, None, None]
            phase_series = self._fuse(phase_fp32, relation_delta)

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
            disagreement = diagnostics["relation_disagreement"]
            peer_gate = (
                (disagreement.median(dim=1, keepdim=True).values
                 / disagreement.clamp_min(1e-6)).clamp(max=1.0)
                if self.use_peer_gate else torch.ones_like(disagreement)
            )
            window_disagreement = diagnostics["source_peer_disagreement"].view(-1, 1)
            source_gate = (
                (self.source_relation_threshold / window_disagreement.clamp_min(1e-6)).clamp(max=1.0)
                if self.use_source_disagreement_gate else torch.ones_like(window_disagreement)
            )
            diagnostics = {
                **diagnostics,
                "source_relation_threshold": torch.as_tensor(self.source_relation_threshold, device=history.device),
                "support_scale": torch.as_tensor(self.support_scale, device=history.device),
                "relation_delta_abs_mean": relation_delta.detach().abs().mean(),
                "gated_raw_delta_abs_mean_before_centering": gated_before,
                "gated_raw_delta_abs_mean_after_centering": raw_delta.detach().abs().mean(),
                "scale_gate": scale_gate.detach(),
                "peer_gate": peer_gate.detach(),
                "source_gate": source_gate.detach(),
                "scale_gate_mean": scale_gate.detach().mean(),
                "relation_trust_mean": trust_gate.detach().mean(),
                **self._fusion_diagnostics(phase_fp32, phase_series),
            }
            return forecast, diagnostics
        return forecast


class ConcatProjectAdapter(_DiagnosticControlAdapter):
    """Trainable [Z,Delta] fusion after canonical gating and recentering.

    Both W_X and W_Delta are trainable; direct-path updates need not remain
    bounded by the relation gates. There is no activation or extra centering.
    """

    def _fuse(self, phase: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((phase, delta), dim=-1))

    def _fusion_diagnostics(self, phase, fused):
        return {
            "fusion_x_identity_deviation": (
                self.fusion.weight[:, :7] - torch.eye(7, device=phase.device)
            ).detach().norm(),
            "fusion_delta_weight_norm": self.fusion.weight[:, 7:].detach().norm(),
            "fusion_update_abs_mean": (fused - phase).detach().abs().mean(),
        }


def build_model(variant: str, n_sites: int, threshold: float, seed: int = 13) -> nn.Module:
    """Construct seeded canonical Full first; isolate CPU construction RNG.

    The trainer re-seeds independently. No CUDA device is initialized or its
    RNG touched here. Shared/relation state with matching shape retains the
    exact canonical initialization, irrespective of added branch capacity.
    Metadata is plain data, adding no parameters or persistent state keys.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected {VARIANTS}")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = pcgrad.v21._build_model(n_sites, float(threshold), support_scale=1.0).float()
        if variant == "phase_shared_strict":
            model.__class__ = _DiagnosticControlAdapter
            model.relation.__class__ = PhaseSharedStrictRelation
            model.relation.mode = variant
        elif variant == "stationwise_matched":
            model.__class__ = _DiagnosticControlAdapter
            model.relation = StationwiseMatchedRelation(model.relation.cycles)
        elif variant == "concat_project":
            model.__class__ = ConcatProjectAdapter
            model.fusion = nn.Linear(14, 7)
            with torch.no_grad():
                model.fusion.weight.copy_(torch.cat((torch.eye(7), torch.eye(7)), dim=1))
                model.fusion.bias.zero_()
    model.variant = variant
    model.variant_metadata = {
        "variant": variant, "version": "supplementary_design_controls_20261117_v1",
        "seed": seed, "relation_mode": model.relation.mode,
        "canonical_builder": "pcgrad.v21._build_model",
        "shared_normalized_operator": variant == "phase_shared_strict",
        "learned_cross_station_aggregation": variant != "stationwise_matched",
        "shared_deterministic_centering_and_attenuation": True,
        "peer_gate_mathematically_one": variant == "stationwise_matched",
        "trainable_direct_fusion_path": variant == "concat_project",
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model


__all__ = ["VARIANTS", "VARIANTS4", "build_model"]

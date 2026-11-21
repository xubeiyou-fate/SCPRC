"""Inference/training-shape adapter for the pinned official PhaseFormer core.

The official model inherits a Lightning base class, but its forecasting
``forward`` uses ordinary PyTorch modules.  When Lightning is unavailable,
this independent runner supplies only the two base-class methods needed to
instantiate that unchanged forward implementation.  Lightning training and
its epoch-wise test callback are never used.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "PhaseFormer"
UPSTREAM_COMMIT = "ff14c16fab136aef60048b20f2ef0d3bc6d86fc7"


def _load_official_model():
    # Other lightweight PhaseFormer wrappers may already have installed a
    # minimal compatibility module in sys.modules. find_spec() raises
    # ValueError for that module because it has no __spec__, so only query it
    # when no module is present and treat an absent/broken spec as unavailable.
    lightning_available = "pytorch_lightning" in sys.modules
    if not lightning_available:
        try:
            lightning_available = importlib.util.find_spec("pytorch_lightning") is not None
        except (ImportError, ValueError):
            lightning_available = False
    if not lightning_available:
        compatibility = types.ModuleType("pytorch_lightning")

        class LightningModule(nn.Module):
            def save_hyperparameters(self, *args, **kwargs):
                return None

        compatibility.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = compatibility
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return importlib.import_module("src.models.PhaseFormer").PhaseFormer


_OfficialPhaseFormer = _load_official_model()


class _Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class PhaseFormerUrbanEV(nn.Module):
    """Use the unchanged official PhaseFormer forward on UrbanEV windows."""

    def __init__(self, seq_len: int = 168, pred_len: int = 24, channels: int = 275,
                 period_len: int = 24, latent_dim: int = 32, phase_layers: int = 2,
                 routers: int = 1, heads: int = 8, dropout: float = 0.1,
                 pos_embed: bool = True):
        super().__init__()
        if seq_len < 1 or pred_len < 1 or channels < 1 or period_len < 1:
            raise ValueError("sequence, horizon, channels and period must be positive")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by attention heads")
        self.seq_len, self.pred_len, self.channels = int(seq_len), int(pred_len), int(channels)
        config = _Config(
            seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.channels,
            period_len=int(period_len), target_var_index=-1,
            latent_dim=int(latent_dim), phase_layers=int(phase_layers),
            phase_encoder_hidden=64, predictor_hidden=128,
            phase_attn_heads=int(heads), phase_attn_dropout=float(dropout),
            phase_attn_use_relpos=True, phase_attn_window=None,
            phase_attention_dim=None, phase_num_routers=int(routers),
            phase_use_pos_embed=bool(pos_embed), phase_pos_dropout=0.0,
            use_revin=True, revin_affine=False, revin_eps=1e-5,
            use_huber_loss=False, huber_delta=1.0,
            training_args=_Config(loss_func="mse", learning_rate=1e-3),
        )
        self.network = _OfficialPhaseFormer(config)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"expected [batch,{self.seq_len},{self.channels}], got {tuple(history.shape)}"
            )
        forecast, _, _ = self.network(history)
        if forecast.shape != (history.shape[0], self.pred_len, self.channels):
            raise RuntimeError(f"unexpected PhaseFormer forecast shape {tuple(forecast.shape)}")
        return forecast


__all__ = ["PhaseFormerUrbanEV", "UPSTREAM_COMMIT"]

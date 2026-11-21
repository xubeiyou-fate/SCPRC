"""Pinned official PatchTST supervised forecasting forward on UrbanEV windows."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parents[1] / "vendor/PatchTST/PatchTST_supervised"
UPSTREAM_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def _official_model():
    # The shared TimeMixer trainer has already imported its own `layers`
    # package.  PatchTST's additional layer modules have distinct names; add
    # only their directory to the existing package search path.  Model code
    # and forward remain unchanged.
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import layers

    path = str(VENDOR / "layers")
    if path not in layers.__path__:
        layers.__path__.append(path)
    spec = importlib.util.spec_from_file_location(
        "urbanev_official_patchtst", VENDOR / "models/PatchTST.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned official PatchTST model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Model


_OfficialModel = _official_model()


class PatchTSTUrbanEV(nn.Module):
    def __init__(self, seq_len: int = 168, pred_len: int = 24, channels: int = 275,
                 patch_len: int = 16, stride: int = 8, d_model: int = 32,
                 d_ff: int = 64, e_layers: int = 1, n_heads: int = 4,
                 dropout: float = 0.1, fc_dropout: float = 0.1,
                 head_dropout: float = 0.0, revin: bool = True,
                 decomposition: bool = False):
        super().__init__()
        if min(seq_len, pred_len, channels, patch_len, stride, d_model, d_ff,
               e_layers, n_heads) <= 0 or d_model % n_heads:
            raise ValueError("invalid PatchTST dimensions")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        config = SimpleNamespace(
            enc_in=self.channels, seq_len=self.seq_len, pred_len=self.pred_len,
            e_layers=int(e_layers), n_heads=int(n_heads), d_model=int(d_model),
            d_ff=int(d_ff), dropout=float(dropout),
            fc_dropout=float(fc_dropout), head_dropout=float(head_dropout),
            individual=False, patch_len=int(patch_len), stride=int(stride),
            padding_patch="end", revin=int(revin), affine=0,
            subtract_last=0, decomposition=int(decomposition), kernel_size=25,
        )
        self.network = _OfficialModel(config)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"expected [batch,{self.seq_len},{self.channels}], got {tuple(history.shape)}")
        forecast = self.network(history)
        if forecast.shape != (history.shape[0], self.pred_len, self.channels):
            raise RuntimeError(f"unexpected PatchTST forecast shape {tuple(forecast.shape)}")
        return forecast


__all__ = ["PatchTSTUrbanEV", "UPSTREAM_COMMIT"]

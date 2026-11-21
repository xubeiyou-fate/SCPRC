"""DLinear/NLinear reference implementations with the shared research trainer.

Architecture reference: cure-lab/LTSF-Linear at commit
0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6, models/DLinear.py and NLinear.py.
The module below is an explicit local reference implementation, rather than a
vendored model.  It retains the source's parameter naming, initialization and
forward equations, so source weights can be loaded for equivalence checks.
Default weights are shared across channels (individual=False), as in LTSF.
Training uses the frozen UrbanEV research adapter, not the LTSF training script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .timemixer import _fit_model


SOURCE_COMMIT = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
SOURCE_BASE = f"https://github.com/cure-lab/LTSF-Linear/blob/{SOURCE_COMMIT}/models"


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("DLinear moving average must have an odd positive kernel")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, edge, 1)
        end = x[:, -1:, :].repeat(1, edge, 1)
        padded = torch.cat((front, x, end), dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class _Decomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = _MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 individual: bool = False, moving_avg: int = 25):
        super().__init__()
        self.seq_len, self.pred_len, self.channels = seq_len, pred_len, channels
        self.individual = individual
        self.decompsition = _Decomposition(moving_avg)
        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            # Preserve the source's alternating initialization order.
            for _ in range(channels):
                self.Linear_Seasonal.append(nn.Linear(seq_len, pred_len))
                self.Linear_Trend.append(nn.Linear(seq_len, pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decompsition(x)
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)
        if self.individual:
            seasonal = torch.stack([layer(seasonal[:, i, :]) for i, layer in
                                    enumerate(self.Linear_Seasonal)], dim=1)
            trend = torch.stack([layer(trend[:, i, :]) for i, layer in
                                 enumerate(self.Linear_Trend)], dim=1)
        else:
            seasonal = self.Linear_Seasonal(seasonal)
            trend = self.Linear_Trend(trend)
        return (seasonal + trend).permute(0, 2, 1)


class NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 individual: bool = False):
        super().__init__()
        self.seq_len, self.pred_len, self.channels = seq_len, pred_len, channels
        self.individual = individual
        if individual:
            self.Linear = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :].detach()
        centered = x - last
        if self.individual:
            forecast = torch.stack([layer(centered[:, :, i]) for i, layer in
                                    enumerate(self.Linear)], dim=2)
        else:
            forecast = self.Linear(centered.permute(0, 2, 1)).permute(0, 2, 1)
        return forecast + last


def fit_predict(config: dict[str, Any], data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Fit DLinear or NLinear on the caller's leakage-safe UrbanEV windows."""
    name = str(config.get("model", "dlinear")).lower()
    seq_len = int(config["seq_len"])
    pred_len = int(config.get("pred_len", np.shape(data["train_y"])[1]))
    channels = np.shape(data["train_x"])[2]
    individual = bool(config.get("individual", False))
    kernel_size = int(config.get("moving_avg", 25))
    if name == "dlinear":
        factory = lambda: DLinear(seq_len, pred_len, channels, individual, kernel_size)
        source_file = "DLinear.py"
    elif name == "nlinear":
        factory = lambda: NLinear(seq_len, pred_len, channels, individual)
        source_file = "NLinear.py"
    else:
        raise ValueError(f"unknown linear baseline {name}")
    metadata = {
        "model_source": "explicit local reference implementation; equations and parameter layout verified against source",
        "reference_commit": SOURCE_COMMIT,
        "reference_url": f"{SOURCE_BASE}/{source_file}",
        "individual": individual,
        "moving_avg": kernel_size if name == "dlinear" else None,
        "future_calendar_features": False,
        "benchmark_scope": "UrbanEV; not scprc of original ETT result tables",
    }
    scale = bool(config.get("scale_aware_loss", False))
    consistency = float(config.get("consistency_weight", 0.0))
    objective = None
    if scale or consistency > 0:
        std = np.asarray(data["scaler_std"], dtype=np.float32)
        median = max(float(np.median(std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        weights = (np.clip((std / median) ** exponent, 0.5, 4.0)
                   if scale else np.ones_like(std))
        weights /= max(float(weights.mean()), 1e-6)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        factors = tuple(int(v.strip()) for v in
                        str(config.get("consistency_factors", "2,3,4,6,12")).split(",")
                        if v.strip())

        def objective(model, x, target):
            pred = model(x)
            w = weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
            point = (w * (pred - target).square()).mean()
            terms = []
            for factor in factors:
                if factor < 2 or pred.size(1) % factor:
                    continue
                p = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(2)
                y = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(2)
                terms.append((w * (p - y).square()).mean())
            return point + consistency * torch.stack(terms).mean() if terms and consistency > 0 else point

        metadata["loss_name"] = ("multi_resolution_scale_consistency" if consistency > 0
                                 else "scale_weighted_mse")
    return _fit_model(config, data, run_dir, factory, lambda model, x: model(x),
                      metadata, loss_fn=objective)


__all__ = ["DLinear", "NLinear", "fit_predict"]

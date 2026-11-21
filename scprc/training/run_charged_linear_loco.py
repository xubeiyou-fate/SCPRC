#!/usr/bin/env python3
"""Matched two-input DLinear/NLinear baselines for CHARGED source-only LOCO.

This runner reuses the frozen CHARGED prefix loader, fold scaler, window
indexing, and batch construction from ``run_charged_covariate_hptn_phaseformer_loco.py``.
It supports only the agreed 512- or 2048-update budgets, batch size 4, Adam
1e-3, standardized-volume MSE, and zero target adaptation. Without ``--run``
or ``--smoke-only`` it does not load data or train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = HERE.parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

# Import the active protocol primitives so these baselines cannot silently
# drift in data prefix, scaler, window boundaries, or source minibatch shape.
from scprc.training.run_charged_covariate_hptn_phaseformer_loco import (  # noqa: E402
    CITY_CODES,
    TRAIN_END,
    VAL_END,
    VARIABLES,
    load_feature_arrays,
    make_batch,
    metrics,
    source_only_scalers,
    window_starts,
)

CONTEXT = 168
HORIZON = 24
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
SUPPORTED_STEPS = (512, 2048)
MOVING_AVG = 25
MODEL_NAMES = ("dlinear", "nlinear")


class _MovingAverage(nn.Module):
    """DLinear endpoint-replicated moving average."""

    def __init__(self, kernel_size: int = MOVING_AVG):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("moving-average kernel must be a positive odd integer")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = (self.kernel_size - 1) // 2
        padded = torch.cat((x[:, :1].expand(-1, edge, -1), x,
                            x[:, -1:].expand(-1, edge, -1)), dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class _DLinear2to1(nn.Module):
    """DLinear temporal decomposition with a learned two-variable readout.

    The original DLinear maps each input channel independently to that same
    output channel. For the requested volume-duration -> volume task, its
    trend/seasonal temporal maps are shared across the two inputs and a small
    shared readout combines the resulting variables at each forecast horizon.
    """

    def __init__(self, seq_len: int = CONTEXT, pred_len: int = HORIZON,
                 channels: int = len(VARIABLES), moving_avg: int = MOVING_AVG):
        super().__init__()
        if channels != len(VARIABLES):
            raise ValueError(f"expected {len(VARIABLES)} inputs ordered as {VARIABLES}")
        self.moving_average = _MovingAverage(moving_avg)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)
        self.trend_projection = nn.Linear(seq_len, pred_len)
        self.variable_readout = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (CONTEXT, len(VARIABLES)):
            raise ValueError(f"expected [batch,{CONTEXT},{len(VARIABLES)}], got {tuple(x.shape)}")
        trend = self.moving_average(x)
        seasonal = x - trend
        seasonal_future = self.seasonal_projection(seasonal.transpose(1, 2)).transpose(1, 2)
        trend_future = self.trend_projection(trend.transpose(1, 2)).transpose(1, 2)
        return self.variable_readout(seasonal_future + trend_future).squeeze(-1)


class _NLinear2to1(nn.Module):
    """NLinear endpoint-centering with a learned two-variable readout."""

    def __init__(self, seq_len: int = CONTEXT, pred_len: int = HORIZON,
                 channels: int = len(VARIABLES)):
        super().__init__()
        if channels != len(VARIABLES):
            raise ValueError(f"expected {len(VARIABLES)} inputs ordered as {VARIABLES}")
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.variable_readout = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (CONTEXT, len(VARIABLES)):
            raise ValueError(f"expected [batch,{CONTEXT},{len(VARIABLES)}], got {tuple(x.shape)}")
        last = x[:, -1:, :].detach()
        centered = x - last
        projected = self.temporal_projection(centered.transpose(1, 2)).transpose(1, 2)
        fused_delta = self.variable_readout(projected).squeeze(-1)
        # Keep NLinear's level restoration anchored to the target (volume) channel.
        return fused_delta + last[..., 0]


def build_model(name: str) -> nn.Module:
    if name == "dlinear":
        return _DLinear2to1()
    if name == "nlinear":
        return _NLinear2to1()
    raise ValueError(f"unsupported baseline: {name}")


def _flatten_batch(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Convert protocol batches to node-as-sample tensors for linear models."""
    if x.ndim != 4 or y.ndim != 3:
        raise ValueError("protocol batches must have shapes [B,T,V,N] and [B,H,N]")
    batch, steps, variables, nodes = x.shape
    if steps != CONTEXT or variables != len(VARIABLES) or y.shape != (batch, HORIZON, nodes):
        raise ValueError("batch dimensions do not match the frozen CHARGED protocol")
    model_x = x.permute(0, 3, 1, 2).contiguous().reshape(batch * nodes, steps, variables)
    model_y = y.permute(0, 2, 1).contiguous().reshape(batch * nodes, HORIZON)
    return model_x, model_y, (batch, nodes)


def _fit_one(name: str, cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...],
             mean: np.ndarray, std: np.ndarray, *, seed: int, steps: int,
             device: str) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(name).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts = window_starts(0, TRAIN_END)
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    loss_sum = 0.0

    for update in range(steps):
        city_id = source_ids[update % len(source_ids)]
        x, y = make_batch(cities[city_id]["values"], starts, mean, std,
                          rng=rng, batch_size=BATCH_SIZE)
        x, y, _ = _flatten_batch(x.to(device), y.to(device))
        model.train()
        prediction = model(x)
        if prediction.shape != y.shape:
            raise RuntimeError(f"{name} prediction {tuple(prediction.shape)} != target {tuple(y.shape)}")
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {name} training loss at update {update}")
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f"nonfinite {name} gradient at update {update}")
        optimizer.step()
        updates_by_city[city_id] += 1
        loss_sum += float(loss.detach().cpu())

    info = {
        "model": name,
        "architecture": (
            "DLinear decomposition with shared per-variable temporal projections and a shared variable-to-volume readout"
            if name == "dlinear" else
            "NLinear endpoint centering with shared per-variable temporal projection, volume-level restoration, and variable-to-volume readout"
        ),
        "seed": int(seed),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "MSE on standardized volume only",
        "batch_size_windows": BATCH_SIZE,
        "source_optimizer_updates_total": int(steps),
        "source_optimizer_updates_by_city": updates_by_city,
        "target_adapt_steps": 0,
        "mean_training_loss": loss_sum / steps,
    }
    training_state = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": rng.bit_generator.state,
    }
    return model, info, training_state


def _predict(model: nn.Module, city: dict[str, Any], mean: np.ndarray,
             std: np.ndarray, *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = city["values"]
    starts = window_starts(TRAIN_END, VAL_END)
    prediction_batches: list[np.ndarray] = []
    truth_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            picks = starts[offset:offset + BATCH_SIZE]
            xs = np.stack([values[s - CONTEXT:s] for s in picks]).astype(np.float32)
            xs = xs.transpose(0, 1, 3, 2)
            xs = (xs - mean.reshape(1, 1, len(VARIABLES), 1)) / std.reshape(1, 1, len(VARIABLES), 1)
            tensor = torch.from_numpy(xs).to(device)
            batch, _, _, nodes = tensor.shape
            model_x = tensor.permute(0, 3, 1, 2).contiguous().reshape(batch * nodes, CONTEXT, len(VARIABLES))
            pred_scaled = model(model_x).reshape(batch, nodes, HORIZON).transpose(1, 2)
            prediction_batches.append(pred_scaled.cpu().numpy())
            truth_batches.append(np.stack([values[s:s + HORIZON, :, 0] for s in picks]).astype(np.float32))
    prediction_scaled = np.concatenate(prediction_batches, axis=0)
    truth_raw = np.concatenate(truth_batches, axis=0)
    prediction_raw = (prediction_scaled * std[0] + mean[0]).astype(np.float32)
    return prediction_raw, truth_raw, starts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_provenance() -> dict[str, str]:
    paths = (
        HERE,
        EXPERIMENTS / "run_charged_covariate_hptn_phaseformer_loco.py",
        ROOT / "scprc/data_adapters/cross_city_adapter.py",
        ROOT / "scprc/data_adapters/charged/city_adapter.py",
        ROOT / "scprc/models/linear.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths if path.is_file()}


def run(*, output_dir: Path, seeds: tuple[int, ...], steps: int, device: str) -> dict[str, Any]:
    if steps not in SUPPORTED_STEPS:
        raise ValueError(f"steps must be one of {SUPPORTED_STEPS}")
    cities, data_provenance = load_feature_arrays()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    prediction_paths: list[str] = []
    state_paths: list[str] = []

    for target_city in CITY_CODES:
        source_ids = tuple(city for city in CITY_CODES if city != target_city)
        mean, std = source_only_scalers(cities, source_ids)
        for seed in seeds:
            fold_predictions: dict[str, np.ndarray] = {}
            fold_truth: np.ndarray | None = None
            label_starts: np.ndarray | None = None
            states: dict[str, Any] = {}
            fold_rows: list[dict[str, Any]] = []
            for name in MODEL_NAMES:
                model, train_info, model_state = _fit_one(
                    name, cities, source_ids, mean, std,
                    seed=seed, steps=steps, device=device,
                )
                prediction, truth, starts = _predict(model, cities[target_city], mean, std, device=device)
                if fold_truth is None:
                    fold_truth, label_starts = truth, starts
                elif not np.array_equal(fold_truth, truth) or not np.array_equal(label_starts, starts):
                    raise AssertionError("DLinear and NLinear predictions use different validation labels")
                fold_predictions[name] = prediction
                row = {
                    **train_info,
                    "target_city": target_city,
                    "source_cities": list(source_ids),
                    "metrics_raw_volume": metrics(prediction, truth),
                    "validation_label_start_index": TRAIN_END,
                    "validation_label_end_exclusive": VAL_END,
                    "validation_windows": int(len(starts)),
                    "site_count": int(len(cities[target_city]["node_ids"])),
                    "source_scaler_mean": mean.tolist(),
                    "source_scaler_std": std.tolist(),
                }
                fold_rows.append(row)
                states[name] = model_state
                del model

            assert fold_truth is not None and label_starts is not None
            prediction_path = output_dir / f"charged_linear_{target_city.lower()}_seed{seed}_predictions.npz"
            np.savez_compressed(
                prediction_path,
                label_starts=label_starts,
                label_timestamps=np.asarray([
                    str(cities[target_city]["times"][int(start)]) for start in label_starts
                ]),
                site_ids=np.asarray(cities[target_city]["node_ids"], dtype=str),
                truth_volume_raw=fold_truth,
                prediction_dlinear_raw=fold_predictions["dlinear"],
                prediction_nlinear_raw=fold_predictions["nlinear"],
            )
            state_path = output_dir / f"charged_linear_{target_city.lower()}_seed{seed}_states.pt"
            torch.save({
                "models": states,
                "fold": {
                    "target_city": target_city,
                    "source_cities": list(source_ids),
                    "seed": int(seed),
                    "steps": int(steps),
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "scaler_mean": mean,
                    "scaler_std": std,
                },
            }, state_path)
            prediction_paths.append(str(prediction_path))
            state_paths.append(str(state_path))
            result_rows.extend(fold_rows)
            print(f"completed target={target_city} seed={seed} steps={steps}", flush=True)

    payload = {
        "experiment": "CHARGED v3 source-only LOCO, two-variable DLinear/NLinear volume forecast",
        "status": "completed",
        "protocol": {
            "cities": list(CITY_CODES),
            "context": CONTEXT,
            "horizon": HORIZON,
            "variables_order": list(VARIABLES),
            "target": "volume",
            "train_end_exclusive": TRAIN_END,
            "validation_label_start": TRAIN_END,
            "validation_end_exclusive": VAL_END,
            "validation_month": "2023-08",
            "validation_windows_per_city": 721,
            "scaler": "exact active-runner source-only equal-city per-variable global mean/std, fit on [0,2928)",
            "target_scaler_data_used": False,
            "target_adaptation_steps": 0,
            "source_optimizer_updates_per_model_target_seed": steps,
            "source_city_update_schedule": "round-robin source cities, one minibatch/update",
            "batch_size_windows": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "MSE on standardized volume only",
            "remove_zero": False,
            "future_covariates_used": False,
            "seeds": list(seeds),
            "device": device,
        },
        "architectures": {
            "dlinear": {
                "reference": "DLinear trend/seasonal decomposition with shared temporal projections",
                "adapter": "shared temporal projections per input variable followed by a shared two-variable-to-volume readout",
                "moving_average_kernel": MOVING_AVG,
            },
            "nlinear": {
                "reference": "NLinear endpoint centering with shared temporal projection",
                "adapter": "per-variable centered temporal projections, shared two-variable delta readout, and last-volume level restoration",
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "data_provenance": data_provenance,
        "code_sha256": _code_provenance(),
        "results": result_rows,
        "prediction_files": prediction_paths,
        "state_files": state_paths,
    }
    summary_path = output_dir / "charged_linear_loco_results.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["summary_json"] = str(summary_path)
    return payload


def cpu_smoke() -> dict[str, Any]:
    """Synthetic forward/backward checks only; does not read CHARGED data."""
    torch.manual_seed(0)
    x = torch.randn(8, CONTEXT, len(VARIABLES), dtype=torch.float32)
    target = torch.randn(8, HORIZON, dtype=torch.float32)
    summaries: dict[str, Any] = {}
    for name in MODEL_NAMES:
        model = build_model(name).cpu().train()
        prediction = model(x)
        if prediction.shape != target.shape:
            raise AssertionError(f"{name} output shape {tuple(prediction.shape)} != {tuple(target.shape)}")
        loss = F.mse_loss(prediction, target)
        loss.backward()
        parameters = list(model.parameters())
        if any(parameter.grad is None for parameter in parameters):
            raise AssertionError(f"{name} has a parameter disconnected from volume loss")
        if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise AssertionError(f"{name} has a nonfinite gradient")
        if not any(bool(torch.count_nonzero(parameter.grad).item()) for parameter in parameters):
            raise AssertionError(f"{name} has no nonzero gradients")
        summaries[name] = {
            "output_shape": list(prediction.shape),
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "gradient_tensors": len(parameters),
            "nonzero_gradient_elements": int(sum(
                torch.count_nonzero(parameter.grad).item() for parameter in parameters
            )),
            "finite_loss": bool(torch.isfinite(loss).item()),
            "cpu_forward_backward": True,
        }

    # Check that a protocol-shaped window batch flattens to node-as-sample
    # without mixing the variable axis or changing its aligned target shape.
    synthetic_x = torch.randn(2, CONTEXT, len(VARIABLES), 3)
    synthetic_y = torch.randn(2, HORIZON, 3)
    flat_x, flat_y, original = _flatten_batch(synthetic_x, synthetic_y)
    if flat_x.shape != (6, CONTEXT, len(VARIABLES)) or flat_y.shape != (6, HORIZON) or original != (2, 3):
        raise AssertionError("protocol batch flattening is inconsistent")
    summaries["protocol_batch"] = {
        "flattened_input_shape": list(flat_x.shape),
        "flattened_target_shape": list(flat_y.shape),
        "original_batch_nodes": list(original),
        "ok": True,
    }
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly run all configured LOCO folds/models")
    parser.add_argument("--smoke-only", action="store_true", help="synthetic CPU shape and gradient checks")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "charged_linear_loco")
    parser.add_argument("--seeds", default="13,17")
    parser.add_argument("--steps", type=int, choices=SUPPORTED_STEPS, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.smoke_only:
        print(json.dumps(cpu_smoke(), indent=2))
        return 0
    if not args.run:
        parser.print_help()
        print("\nNo data was loaded and no training was started. Use --smoke-only for CPU checks or --run to execute.")
        return 0
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    result = run(output_dir=args.output_dir, seeds=seeds, steps=args.steps, device=args.device)
    print(result["summary_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

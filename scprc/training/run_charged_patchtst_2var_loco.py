#!/usr/bin/env python3
"""Strict two-variable PatchTST baseline on the CHARGED LOCO protocol.

This runner is intentionally separate from the older one-variable direct
baseline.  The pinned official PatchTST receives the same ``[volume,
duration]`` historical variables, context 168 and horizon 24 as the matched
CHARGED controls.  Only the volume output channel is supervised and reported.
Training uses source cities only (April--July), exactly 2,048 source updates,
and the target city is materialized only through August for final scoring.
September rows are never read.  Each target/seed fold writes its own
predictions, result JSON, and CPU-readable state under an isolated output
directory; existing experiments are never touched.

The official PatchTST backbone is channel-independent by design.  Therefore
the duration channel is supplied as a real second input and is audited in the
provenance, but the architecture does not contain cross-channel attention.  The
official BatchNorm path can create indirect training-mode batch-statistic
coupling; at evaluation mode the duration-to-volume intervention is zero.  This
is recorded explicitly rather than described as cross-variable fusion.  The
volume-only loss is the same standardized point MSE used by the matched
CHARGED source-only controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(ROOT / "scprc") not in sys.path:
    sys.path.insert(0, str(ROOT / "scprc"))

import scprc.training.run_charged_covariate_hptn_phaseformer_loco as protocol  # noqa: E402
# The pinned adapter appends its official layer directory to the already
# imported shared ``layers`` package.  Importing the repository model first is
# therefore part of the audited namespace setup, not an architectural change.
from scprc.models import timemixer as _timemixer_namespace  # noqa: F401,E402
from scprc.baselines.patchtst_adapter import (  # noqa: E402
    PatchTSTUrbanEV,
    UPSTREAM_COMMIT,
)

CITY_CODES = protocol.CITY_CODES
CONTEXT = protocol.CONTEXT
HORIZON = protocol.HORIZON
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END
VARIABLES = protocol.VARIABLES
SEED_DEFAULTS = (13, 17, 29)
DEFAULT_STEPS = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-3


class TwoVariablePatchTST(nn.Module):
    """Pinned official PatchTST with volume+duration input and volume output."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PatchTSTUrbanEV(
            seq_len=CONTEXT,
            pred_len=HORIZON,
            channels=len(VARIABLES),
            patch_len=16,
            stride=8,
            d_model=32,
            d_ff=64,
            e_layers=1,
            n_heads=4,
            # Match the zero-dropout direct CHARGED controls.  RevIN is the
            # official causal per-window normalization and has no learned
            # affine parameters in this configuration.
            dropout=0.0,
            fc_dropout=0.0,
            head_dropout=0.0,
            revin=True,
            decomposition=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CONTEXT, len(VARIABLES)):
            raise ValueError(
                f"expected [batch,{CONTEXT},{len(VARIABLES)}], got {tuple(x.shape)}"
            )
        forecast = self.backbone(x)
        if forecast.ndim != 3 or tuple(forecast.shape[1:]) != (HORIZON, len(VARIABLES)):
            raise RuntimeError(f"unexpected PatchTST output shape {tuple(forecast.shape)}")
        return forecast[..., 0]


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def _record_gradient_activity(model: nn.Module, connected: dict[str, bool],
                              nonzero: dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            connected[name] = True
            if bool(torch.count_nonzero(parameter.grad.detach()).item()):
                nonzero[name] = True


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance() -> dict[str, str]:
    paths = [
        HERE,
        HERE.parent / "run_charged_covariate_hptn_phaseformer_loco.py",
        ROOT / "scprc/baselines/patchtst_adapter.py",
        ROOT / "scprc/data_adapters/cross_city_adapter.py",
        ROOT / "scprc/data_adapters/charged/city_adapter.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/models/PatchTST.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/PatchTST_layers.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/RevIN.py",
    ]
    return {
        str(path.relative_to(ROOT)): _hash_file(path)
        for path in paths if path.is_file()
    }


def _fit_model(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...],
               mean: np.ndarray, std: np.ndarray, *, seed: int, steps: int,
               batch_size: int, learning_rate: float, device: str,
               torch_threads: int) -> tuple[TwoVariablePatchTST, dict[str, Any]]:
    """Train exactly ``steps`` source minibatch updates; target data is absent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = TwoVariablePatchTST().to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_starts = protocol.window_starts(0, TRAIN_END)
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    connected_seen = {
        name: False for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    nonzero_seen = dict.fromkeys(connected_seen, False)
    losses: list[float] = []
    started = time.perf_counter()

    for update in range(int(steps)):
        city = source_ids[update % len(source_ids)]
        x, y = protocol.make_batch(
            cities[city]["values"], train_starts, mean, std,
            rng=rng, batch_size=batch_size,
        )
        # [B,T,V,N] -> [B*N,T,V], preserving both historical variables.
        batch, context, variables, nodes = x.shape
        if (context, variables) != (CONTEXT, len(VARIABLES)):
            raise RuntimeError(f"unexpected source batch shape {tuple(x.shape)}")
        model_x = x.permute(0, 3, 1, 2).contiguous().reshape(
            batch * nodes, context, variables
        ).to(device)
        model_y = y.permute(0, 2, 1).contiguous().reshape(
            batch * nodes, HORIZON
        ).to(device)
        model.train()
        prediction = model(model_x)
        if prediction.shape != model_y.shape:
            raise RuntimeError(f"prediction {tuple(prediction.shape)} != target {tuple(model_y.shape)}")
        loss = torch.nn.functional.mse_loss(prediction, model_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _record_gradient_activity(model, connected_seen, nonzero_seen)
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        updates_by_city[city] += 1
        losses.append(float(loss.detach().cpu()))

    named = dict(model.named_parameters())
    counts = parameter_summary(model)
    counts.update({
        "active_parameter_elements_connected_to_loss_at_least_once": int(
            sum(named[name].numel() for name, seen in connected_seen.items() if seen)
        ),
        "active_parameter_tensors_connected_to_loss_at_least_once": int(sum(connected_seen.values())),
        "parameter_elements_with_nonzero_gradient_at_least_once": int(
            sum(named[name].numel() for name, seen in nonzero_seen.items() if seen)
        ),
        "parameter_tensors_with_nonzero_gradient_at_least_once": int(sum(nonzero_seen.values())),
    })
    return model, {
        "model": "patchtst_official_two_variable",
        "seed": int(seed),
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "loss": "standardized volume MSE; duration is historical input only",
        "batch_size_windows": int(batch_size),
        "source_optimizer_updates_total": int(steps),
        "source_optimizer_updates_by_city": updates_by_city,
        "target_adapt_steps": 0,
        "mean_training_loss": float(np.mean(losses)) if losses else None,
        "last_training_loss": float(losses[-1]) if losses else None,
        "training_seconds": float(time.perf_counter() - started),
        "parameters": counts,
    }


def _predict_city(model: TwoVariablePatchTST, city: dict[str, Any],
                  mean: np.ndarray, std: np.ndarray, *, batch_size: int,
                  device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = city["values"]
    starts = protocol.window_starts(TRAIN_END, VAL_END)
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            picks = starts[offset:offset + batch_size]
            xs = np.stack([values[s - CONTEXT:s] for s in picks]).astype(np.float32)
            xs = xs.transpose(0, 1, 3, 2)  # [B,T,V,N]
            xs = (xs - mean.reshape(1, 1, len(VARIABLES), 1)) / std.reshape(
                1, 1, len(VARIABLES), 1
            )
            tensor = torch.from_numpy(xs).to(device)
            batch, _, _, nodes = tensor.shape
            model_x = tensor.permute(0, 3, 1, 2).contiguous().reshape(
                batch * nodes, CONTEXT, len(VARIABLES)
            )
            pred_scaled = model(model_x).reshape(batch, nodes, HORIZON).transpose(1, 2)
            predictions.append(pred_scaled.cpu().numpy())
            truths.append(np.stack([values[s:s + HORIZON, :, 0] for s in picks]).astype(np.float32))
    pred_scaled = np.concatenate(predictions, axis=0)
    truth_raw = np.concatenate(truths, axis=0)
    prediction_raw = (pred_scaled * std[0] + mean[0]).astype(np.float32)
    return prediction_raw, truth_raw, starts


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def run_fold(*, target: str, seed: int, output_dir: Path, steps: int,
             batch_size: int, learning_rate: float, device: str,
             torch_threads: int, force: bool = False) -> Path:
    target = str(target).upper()
    if target not in CITY_CODES:
        raise ValueError(f"target must be one of {CITY_CODES}")
    fold_dir = output_dir / f"target_{target.lower()}_seed{int(seed)}"
    result_path = fold_dir / "result.json"
    if result_path.exists() and not force:
        raise FileExistsError(f"existing fold {result_path}; pass --force to replace only this new fold")
    fold_dir.mkdir(parents=True, exist_ok=True)
    cities, data_provenance = protocol.load_feature_arrays()
    source_ids = tuple(city for city in CITY_CODES if city != target)
    mean, std = protocol.source_only_scalers(cities, source_ids)
    model, train_info = _fit_model(
        cities, source_ids, mean, std, seed=int(seed), steps=steps,
        batch_size=batch_size, learning_rate=learning_rate, device=device,
        torch_threads=torch_threads,
    )
    prediction, truth, starts = _predict_city(
        model, cities[target], mean, std, batch_size=batch_size, device=device,
    )
    prediction_file = fold_dir / "predictions.npz"
    np.savez_compressed(
        prediction_file,
        label_starts=starts,
        label_timestamps=np.asarray([str(cities[target]["times"][int(s)]) for s in starts]),
        site_ids=np.asarray(cities[target]["node_ids"], dtype=str),
        truth_volume_raw=truth,
        prediction_patchtst_2var_raw=prediction,
    )
    state_file = fold_dir / "state.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, state_file)
    row: dict[str, Any] = {
        **train_info,
        "experiment": "CHARGED strict two-variable PatchTST source-only LOCO",
        "status": "completed",
        "target_city": target,
        "source_cities": list(source_ids),
        "metrics_raw_volume": _metrics(prediction, truth),
        "validation_label_start_index": TRAIN_END,
        "validation_label_end_exclusive": VAL_END,
        "validation_windows": int(len(starts)),
        "site_count": int(len(cities[target]["node_ids"])),
        "prediction_file": str(prediction_file),
        "state_file": str(state_file),
        "protocol": {
            "cities": list(CITY_CODES),
            "context": CONTEXT,
            "horizon": HORIZON,
            "variables_order": list(VARIABLES),
            "target": "volume",
            "train_end_exclusive": TRAIN_END,
            "train_end_timestamp_exclusive": "2023-08-01 00:00:00",
            "validation_month": "2023-08",
            "validation_windows_per_city": 721,
            "scaler": "source-only global mean/std per variable; equal source-city weight; fitted on [0,2928)",
            "target_scaler_data_used": False,
            "target_adaptation_steps": 0,
            "source_optimizer_updates": int(steps),
            "source_city_update_schedule": "round-robin source cities, one minibatch/update",
            "batch_size_windows": int(batch_size),
            "loss": "standardized volume MSE",
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "future_covariates_used": False,
            "september_test_values_materialized": False,
            "source_only": True,
        },
        "architecture": {
            "name": "pinned official PatchTST, two-variable input",
            "implementation": "PatchTST_supervised pinned adapter",
            "upstream_commit": UPSTREAM_COMMIT,
            "input_channels": len(VARIABLES),
            "input_variables": list(VARIABLES),
            "output_channels": len(VARIABLES),
            "reported_output": "channel 0 volume only",
            "patch_len": 16,
            "stride": 8,
            "d_model": 32,
            "d_ff": 64,
            "e_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "revin": True,
            "channel_independence": True,
            "cross_variable_fusion": "none (official channel-independent PatchTST)",
            "normalization": "official RevIN plus BatchNorm encoder default",
            "duration_to_volume_gradient": "zero in eval mode; training-mode BatchNorm may induce indirect batch-statistic coupling",
        },
        "data_provenance": data_provenance,
        "code_provenance": code_provenance(),
    }
    result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result_path


def cpu_smoke() -> dict[str, Any]:
    """Synthetic shape/backward and two-input-path check; no CHARGED rows read."""
    torch.manual_seed(0)
    model = TwoVariablePatchTST().cpu()
    x = torch.randn(3, CONTEXT, len(VARIABLES), dtype=torch.float32, requires_grad=True)
    target = torch.randn(3, HORIZON, dtype=torch.float32)
    prediction = model(x)
    if tuple(prediction.shape) != (3, HORIZON):
        raise AssertionError(f"unexpected output shape {tuple(prediction.shape)}")
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    duration_grad = x.grad[..., 1]
    result = {
        **parameter_summary(model),
        "output_shape": list(prediction.shape),
        "cpu_backward": True,
        "duration_input_gradient_nonzero_elements": int(torch.count_nonzero(duration_grad).item()),
        "official_commit": UPSTREAM_COMMIT,
        "input_channels": len(VARIABLES),
        "channel_independence": True,
    }
    del model
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CITY_CODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(cpu_smoke(), indent=2))
        return 0
    if args.target is None or args.seed is None:
        parser.error("--target and --seed are required unless --smoke-only is used")
    if args.steps != DEFAULT_STEPS:
        parser.error(f"strict P0 baseline fixes --steps={DEFAULT_STEPS}")
    path = run_fold(
        target=args.target, seed=args.seed, output_dir=args.output_dir,
        steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device,
        torch_threads=args.torch_threads, force=args.force,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

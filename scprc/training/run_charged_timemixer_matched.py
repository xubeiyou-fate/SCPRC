#!/usr/bin/env python3
"""Fair volume-only TimeMixer audit under the frozen SCPRC protocol.

The runner reads ``scpcr_matched_protocol_20261023.json`` and fails closed if
the protocol is changed.  One source minibatch is one city batch of four
windows.  A physical micro-batch of one window with four-way gradient
accumulation is the default memory-safe implementation, so every parameter
update still sees exactly four source windows.  The runner keeps source
minibatches, effective windows, and parameter updates as separate quantities.
August is evaluated retrospectively and September is never materialized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = HERE.parent
FROZEN_PROTOCOL = EXPERIMENTS / 'scprc_matched_protocol.json'
DEFAULT_OUTPUT_ROOT = EXPERIMENTS / "charged_timemixer_matched_20261023"

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as volume_protocol  # noqa: E402
from scprc.models.timemixer import _UpstreamModel, _config_namespace  # noqa: E402

EXPECTED_CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
EXPECTED_SEEDS = (13, 17, 29)
EXPECTED_CONTEXT = 168
EXPECTED_HORIZON = 24
EXPECTED_SOURCE_MINIBATCHES = 2048
EXPECTED_BATCH_WINDOWS = 4
EXPECTED_EFFECTIVE_WINDOWS = EXPECTED_SOURCE_MINIBATCHES * EXPECTED_BATCH_WINDOWS
DEFAULT_MICROBATCH_WINDOWS = 1
DEFAULT_INFERENCE_BATCH_WINDOWS = 1
DEFAULT_TORCH_THREADS = 2


def load_frozen_protocol() -> dict[str, Any]:
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    errors: list[str] = []
    inputs = protocol.get("input", {})
    split = protocol.get("data_split", {})
    budget = protocol.get("training_budget", {})
    if inputs.get("features") != ["volume"]:
        errors.append("input.features must be ['volume']")
    if inputs.get("target") != "volume" or inputs.get("duration_input") is not False:
        errors.append("the frozen input must be volume-only")
    if inputs.get("context_hours") != EXPECTED_CONTEXT or inputs.get("forecast_horizon_hours") != EXPECTED_HORIZON:
        errors.append("context/horizon must be 168/24")
    if tuple(split.get("cities", ())) != EXPECTED_CITIES:
        errors.append("city list does not match the six-city protocol")
    if budget.get("source_minibatches_total") != EXPECTED_SOURCE_MINIBATCHES:
        errors.append("training_budget.source_minibatches_total must be 2048")
    if budget.get("batch_size_windows") != EXPECTED_BATCH_WINDOWS:
        errors.append("training_budget.batch_size_windows must be 4")
    if budget.get("effective_source_windows") != EXPECTED_EFFECTIVE_WINDOWS:
        errors.append("training_budget.effective_source_windows must be 8192")
    if budget.get("single_task_parameter_updates") != EXPECTED_SOURCE_MINIBATCHES:
        errors.append("training_budget.single_task_parameter_updates must be 2048")
    if tuple(budget.get("seeds", ())) != EXPECTED_SEEDS:
        errors.append("seed list must be 13,17,29")
    if budget.get("parameter_optimizer") != "AdamW":
        errors.append("parameter optimizer must be AdamW")
    if budget.get("weight_decay") != 0.0001:
        errors.append("weight decay must be 0.0001")
    if errors:
        raise ValueError("frozen SCPRC protocol validation failed: " + "; ".join(errors))
    return protocol


class MatchedVolumeTimeMixer(nn.Module):
    """Official vendored TimeMixer configured for one volume channel."""

    def __init__(self) -> None:
        super().__init__()
        config = {
            "model": "timemixer_cd",
            "protocol": "research_volume",
            "seq_len": EXPECTED_CONTEXT,
            "pred_len": EXPECTED_HORIZON,
            "d_model": 64,
            "d_ff": 128,
            "e_layers": 2,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "moving_avg": 25,
            "dropout": 0.1,
            "channel_independence": 0,
            "use_norm": 1,
        }
        args = _config_namespace(config, channels=1, pred_len=EXPECTED_HORIZON)
        if int(args.enc_in) != 1 or int(args.channel_independence) != 0:
            raise AssertionError("matched TimeMixer must be one-channel CD route")
        self.backbone = _UpstreamModel(args)
        self.config = args

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (EXPECTED_CONTEXT, 1):
            raise ValueError(f"expected [batch,{EXPECTED_CONTEXT},1], got {tuple(history.shape)}")
        forecast = self.backbone(history, None, None, None)
        if forecast.ndim != 3 or tuple(forecast.shape[1:]) != (EXPECTED_HORIZON, 1):
            raise RuntimeError(f"unexpected TimeMixer output shape {tuple(forecast.shape)}")
        return forecast[..., 0]


def parameter_summary(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _flatten_windows(history: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if history.ndim != 3 or target.ndim != 3:
        raise ValueError("volume protocol batches must be [windows,time,sites] and [windows,horizon,sites]")
    windows, context, sites = history.shape
    if context != EXPECTED_CONTEXT or target.shape != (windows, EXPECTED_HORIZON, sites):
        raise ValueError("batch does not match the frozen 168-to-24 volume protocol")
    model_history = history.permute(0, 2, 1).contiguous().reshape(windows * sites, context, 1)
    model_target = target.permute(0, 2, 1).contiguous().reshape(windows * sites, EXPECTED_HORIZON)
    return model_history, model_target


def _objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    point = (prediction - target).square().mean()
    terms = []
    for factor in (2, 3, 4, 6, 12):
        pred_scale = prediction.reshape(
            prediction.shape[0], prediction.shape[1] // factor, factor, prediction.shape[2]
        ).mean(dim=2)
        target_scale = target.reshape(
            target.shape[0], target.shape[1] // factor, factor, target.shape[2]
        ).mean(dim=2)
        terms.append((pred_scale - target_scale).square().mean())
    return point + 0.15 * torch.stack(terms).mean()


def _device_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "current_allocated_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
    }


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance() -> dict[str, str]:
    paths = [
        HERE,
        FROZEN_PROTOCOL,
        EXPERIMENTS / "training_protocol.py",
        ROOT / "scprc/models/timemixer.py",
    ]
    paths.extend(sorted((ROOT / "scprc/vendor/TimeMixer/models").rglob("*.py")))
    paths.extend(sorted((ROOT / "scprc/vendor/TimeMixer/layers").rglob("*.py")))
    return {str(path.relative_to(ROOT)): _hash_file(path) for path in paths if path.is_file()}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_fold_config(fold_dir: Path, protocol: dict[str, Any], *, target: str, seed: int,
                      microbatch_windows: int, inference_batch_windows: int,
                      torch_threads: int, device: str) -> None:
    budget = {
        "source_minibatches_total": EXPECTED_SOURCE_MINIBATCHES,
        "batch_windows_per_source_minibatch": EXPECTED_BATCH_WINDOWS,
        "effective_source_windows": EXPECTED_EFFECTIVE_WINDOWS,
        "physical_microbatch_windows": microbatch_windows,
        "gradient_accumulation_steps": EXPECTED_BATCH_WINDOWS // microbatch_windows,
        "parameter_updates_total": EXPECTED_SOURCE_MINIBATCHES,
        "parameter_update_semantics": "one TimeMixer optimizer.step after one source minibatch of four windows",
        "pcgrad_comparison_note": "PCGrad five-source training aggregates approximately 410 city-gradient rounds; this single-model TimeMixer baseline performs 2048 parameter updates. The matched unit is source minibatches/effective windows, not optimizer-update count.",
    }
    _write_json(fold_dir / "protocol.json", {
        "frozen_protocol_file": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "frozen_protocol": protocol,
        "runner": {
            "model": "official vendored TimeMixer channel-dependent route",
            "input": "volume-only",
            "target": "volume",
            "target_city_adaptation_steps": 0,
            "august_role": "retrospective evaluation only",
            "september_values_materialized": False,
            "target_city": target,
            "seed": seed,
            "microbatch_windows": microbatch_windows,
            "inference_batch_windows": inference_batch_windows,
            "torch_threads": torch_threads,
            "device": device,
        },
        "budget_accounting": budget,
    })


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if torch.cuda.is_available():
        result.update({
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        })
    return result


def fit_model(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...], mean: float, std: float,
              *, seed: int, source_minibatches: int, batch_windows: int,
              microbatch_windows: int, learning_rate: float, weight_decay: float,
              device: torch.device, torch_threads: int, progress_every: int) -> tuple[nn.Module, dict[str, Any]]:
    if batch_windows != EXPECTED_BATCH_WINDOWS:
        raise ValueError("the frozen matched protocol requires batch_windows=4")
    if microbatch_windows < 1 or batch_windows % microbatch_windows:
        raise ValueError("microbatch_windows must divide batch_windows")
    _seed_everything(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = MatchedVolumeTimeMixer().float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    starts = volume_protocol._train_starts()
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    windows_by_city = {city: 0 for city in source_ids}
    losses: list[float] = []
    accumulation_steps = batch_windows // microbatch_windows
    started = time.perf_counter()
    model.train()
    for source_minibatch_index in range(source_minibatches):
        city = source_ids[source_minibatch_index % len(source_ids)]
        history, target = volume_protocol._make_batch(
            cities[city]["values"][:volume_protocol.TRAIN_END], starts, mean, std,
            rng, batch_windows,
        )
        history, target = history.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        minibatch_loss = 0.0
        for offset in range(0, batch_windows, microbatch_windows):
            physical_history = history[offset:offset + microbatch_windows]
            physical_target = target[offset:offset + microbatch_windows]
            model_history, _ = _flatten_windows(physical_history, physical_target)
            prediction = model(model_history)
            windows = physical_history.shape[0]
            sites = physical_history.shape[2]
            prediction_windows = prediction.reshape(windows, sites, EXPECTED_HORIZON).transpose(1, 2)
            loss = _objective(prediction_windows.float(), physical_target.float())
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TimeMixer loss at source minibatch {source_minibatch_index + 1}")
            (loss / accumulation_steps).backward()
            minibatch_loss += float(loss.detach().cpu()) * physical_history.shape[0] / batch_windows
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite TimeMixer gradient at source minibatch {source_minibatch_index + 1}")
        optimizer.step()
        losses.append(minibatch_loss)
        updates_by_city[city] += 1
        windows_by_city[city] += batch_windows
        if progress_every and (source_minibatch_index + 1) % progress_every == 0:
            print(f"seed={seed} minibatch={source_minibatch_index + 1}/{source_minibatches} city={city} loss={minibatch_loss:.6f}", flush=True)
    info = {
        "source_minibatches_total": int(source_minibatches),
        "source_minibatches_by_city": updates_by_city,
        "effective_source_windows": int(source_minibatches * batch_windows),
        "effective_source_windows_by_city": windows_by_city,
        "parameter_updates_total": int(source_minibatches),
        "parameter_updates_by_city": dict(updates_by_city),
        "optimizer_step_count": int(source_minibatches),
        "batch_windows_per_source_minibatch": int(batch_windows),
        "physical_microbatch_windows": int(microbatch_windows),
        "gradient_accumulation_steps": int(accumulation_steps),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "loss": "standardized volume MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "gradient_aggregation": "single-model averaged gradient over four windows; no PCGrad task projection",
        "pcgrad_comparison_note": "PCGrad five-source protocol uses approximately 410 parameter-update rounds; this TimeMixer baseline uses 2048 parameter updates. Exposure is matched by 2048 source minibatches and 8192 windows.",
        "mean_training_loss": float(np.mean(losses)),
        "training_seconds": float(time.perf_counter() - started),
        "parameters": parameter_summary(model),
    }
    return model, info


def predict_city(model: nn.Module, city: dict[str, Any], mean: float, std: float,
                 *, device: torch.device, batch_windows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(volume_protocol.TRAIN_END, volume_protocol.VAL_END - EXPECTED_HORIZON + 1, dtype=np.int64)
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    values = city["values"]
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_windows):
            picks = starts[offset:offset + batch_windows]
            history = np.stack([values[start - EXPECTED_CONTEXT:start] for start in picks]).astype(np.float32)
            target = np.stack([values[start:start + EXPECTED_HORIZON] for start in picks]).astype(np.float32)
            history_tensor = torch.from_numpy((history - mean) / std).to(device)
            model_history, _ = _flatten_windows(history_tensor, torch.from_numpy((target - mean) / std))
            prediction = model(model_history)
            sites = history.shape[2]
            prediction_scaled = prediction.reshape(len(picks), sites, EXPECTED_HORIZON).transpose(1, 2)
            predictions.append((prediction_scaled.cpu().numpy() * std + mean).astype(np.float32))
            truths.append(target)
    return np.concatenate(predictions, axis=0), np.concatenate(truths, axis=0), starts


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error)))}


def _write_failure(fold_dir: Path, *, protocol: dict[str, Any], target: str, seed: int,
                   device: torch.device, error: BaseException, status: str,
                   **config: Any) -> dict[str, Any]:
    result = {
        "status": status,
        "experiment": "CHARGED source-only LOCO official TimeMixer matched 20261023",
        "target_city": target,
        "seed": seed,
        "source_minibatches_total": EXPECTED_SOURCE_MINIBATCHES,
        "effective_source_windows": EXPECTED_EFFECTIVE_WINDOWS,
        "parameter_updates_total": EXPECTED_SOURCE_MINIBATCHES,
        "budget_accounting": "TimeMixer performs one parameter update per four-window source minibatch; PCGrad five-source rounds are not numerically interchangeable with this count.",
        "config": config,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "environment": _environment(device),
        "code_provenance": code_provenance(),
    }
    _write_json(fold_dir / "result.json", result)
    return result


def run_fold(cities: dict[str, dict[str, Any]], data_provenance: dict[str, Any], protocol: dict[str, Any],
             *, target: str, seed: int, output_root: Path, device_name: str,
             microbatch_windows: int, inference_batch_windows: int, torch_threads: int,
             progress_every: int) -> dict[str, Any]:
    fold_dir = output_root / f"target_{target.lower()}_seed{seed}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    _write_fold_config(
        fold_dir, protocol, target=target, seed=seed,
        microbatch_windows=microbatch_windows,
        inference_batch_windows=inference_batch_windows,
        torch_threads=torch_threads, device=device_name,
    )
    (fold_dir / "command.txt").write_text(" ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8")
    _write_json(fold_dir / "environment.json", _environment(device))
    started = time.perf_counter()
    try:
        source_ids = tuple(city for city in EXPECTED_CITIES if city != target)
        mean, std = volume_protocol.source_scaler(cities, source_ids)
        _reset_memory(device)
        model, training = fit_model(
            cities, source_ids, mean, std,
            seed=seed, source_minibatches=EXPECTED_SOURCE_MINIBATCHES,
            batch_windows=EXPECTED_BATCH_WINDOWS, microbatch_windows=microbatch_windows,
            learning_rate=0.001, weight_decay=0.0001, device=device,
            torch_threads=torch_threads, progress_every=progress_every,
        )
        train_memory = _device_memory(device)
        checkpoint_path = fold_dir / "trained_checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "model": "official TimeMixer volume-only", "target_city": target, "seed": seed}, checkpoint_path)
        _reset_memory(device)
        inference_started = time.perf_counter()
        prediction, truth, starts = predict_city(
            model, cities[target], mean, std,
            device=device, batch_windows=inference_batch_windows,
        )
        inference_seconds = time.perf_counter() - inference_started
        inference_memory = _device_memory(device)
        metrics = score(prediction, truth)
        prediction_path = fold_dir / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            label_starts=starts,
            label_timestamps=np.asarray([str(cities[target]["times"][int(start)]) for start in starts]),
            site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
            truth_raw=truth,
            prediction_timemixer_volume_raw=prediction,
        )
        result = {
            "status": "completed",
            "experiment": "CHARGED source-only LOCO official TimeMixer matched 20261023",
            "target_city": target,
            "source_cities": list(source_ids),
            "seed": seed,
            "protocol": {
                "input": "volume-only",
                "target": "volume",
                "context_hours": EXPECTED_CONTEXT,
                "horizon_hours": EXPECTED_HORIZON,
                "source_scaler": "equal-city-weight source-only volume mean/std over indices [0,2928)",
                "august_role": "retrospective evaluation only",
                "september_values_materialized": False,
                "validation_origins": int(len(starts)),
                "target_adaptation_steps": 0,
            },
            "source_minibatches_total": training["source_minibatches_total"],
            "source_minibatches_by_city": training["source_minibatches_by_city"],
            "effective_source_windows": training["effective_source_windows"],
            "effective_source_windows_by_city": training["effective_source_windows_by_city"],
            "parameter_updates_total": training["parameter_updates_total"],
            "parameter_updates_by_city": training["parameter_updates_by_city"],
            "optimizer_step_count": training["optimizer_step_count"],
            "training": training,
            "metrics_raw_volume": metrics,
            "training_memory": train_memory,
            "inference_memory": inference_memory,
            "inference_batch_windows": inference_batch_windows,
            "inference_seconds": float(inference_seconds),
            "training_and_inference_seconds": float(time.perf_counter() - started),
            "checkpoint": str(checkpoint_path),
            "predictions": str(prediction_path),
            "data_provenance": data_provenance,
            "code_provenance": code_provenance(),
        }
        _write_json(fold_dir / "result.json", result)
        del model
        _reset_memory(device)
        return result
    except torch.cuda.OutOfMemoryError as error:
        _reset_memory(device)
        return _write_failure(
            fold_dir, protocol=protocol, target=target, seed=seed, device=device,
            error=error, status="oom",
            microbatch_windows=microbatch_windows,
            inference_batch_windows=inference_batch_windows,
            source_minibatches_total=EXPECTED_SOURCE_MINIBATCHES,
            effective_source_windows=EXPECTED_EFFECTIVE_WINDOWS,
            parameter_updates_total=EXPECTED_SOURCE_MINIBATCHES,
        )
    except Exception as error:
        _reset_memory(device)
        return _write_failure(
            fold_dir, protocol=protocol, target=target, seed=seed, device=device,
            error=error, status="failed",
            microbatch_windows=microbatch_windows,
            inference_batch_windows=inference_batch_windows,
            source_minibatches_total=EXPECTED_SOURCE_MINIBATCHES,
            effective_source_windows=EXPECTED_EFFECTIVE_WINDOWS,
            parameter_updates_total=EXPECTED_SOURCE_MINIBATCHES,
        )


def _summary_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("target_*_seed*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_summary(output_root: Path, protocol: dict[str, Any], rows: list[dict[str, Any]], *, status: str) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    metrics = [row["metrics_raw_volume"] for row in completed if "metrics_raw_volume" in row]
    summary = {
        "status": status,
        "experiment": "CHARGED source-only LOCO official TimeMixer matched 20261023",
        "protocol_file": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "budget_accounting": {
            "source_minibatches_total": EXPECTED_SOURCE_MINIBATCHES,
            "batch_windows_per_source_minibatch": EXPECTED_BATCH_WINDOWS,
            "effective_source_windows": EXPECTED_EFFECTIVE_WINDOWS,
            "parameter_updates_total_per_single_model_fold": EXPECTED_SOURCE_MINIBATCHES,
            "pcgrad_reference": "five-source PCGrad has approximately 410 parameter-update rounds; do not equate that count with this single-model 2048 count",
        },
        "fold_count": len(rows),
        "completed_count": len(completed),
        "status_counts": {value: sum(row.get("status") == value for row in rows) for value in sorted({row.get("status") for row in rows})},
        "mean_completed_rmse": float(np.mean([value["rmse"] for value in metrics])) if metrics else None,
        "mean_completed_mae": float(np.mean([value["mae"] for value in metrics])) if metrics else None,
        "results": rows,
        "frozen_protocol": protocol,
    }
    _write_json(output_root / "summary.json", summary)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "target_city", "seed", "status", "rmse", "mae",
            "source_minibatches_total", "effective_source_windows", "parameter_updates_total",
            "training_seconds", "inference_seconds", "train_peak_allocated_bytes", "inference_peak_allocated_bytes",
            "error_type",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "target_city": row.get("target_city"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "rmse": row.get("metrics_raw_volume", {}).get("rmse"),
                "mae": row.get("metrics_raw_volume", {}).get("mae"),
                "source_minibatches_total": row.get("source_minibatches_total"),
                "effective_source_windows": row.get("effective_source_windows"),
                "parameter_updates_total": row.get("parameter_updates_total"),
                "training_seconds": row.get("training", {}).get("training_seconds"),
                "inference_seconds": row.get("inference_seconds"),
                "train_peak_allocated_bytes": row.get("training_memory", {}).get("peak_allocated_bytes"),
                "inference_peak_allocated_bytes": row.get("inference_memory", {}).get("peak_allocated_bytes"),
                "error_type": row.get("error_type"),
            })


def run_all(args: argparse.Namespace) -> int:
    protocol = load_frozen_protocol()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.microbatch_windows < 1 or EXPECTED_BATCH_WINDOWS % args.microbatch_windows:
        raise ValueError("--microbatch-windows must divide 4")
    if args.inference_batch_windows < 1:
        raise ValueError("--inference-batch-windows must be positive")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "protocol.json", {
        "frozen_protocol_file": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "frozen_protocol": protocol,
        "runner_budget": {
            "source_minibatches_total": EXPECTED_SOURCE_MINIBATCHES,
            "batch_windows_per_source_minibatch": EXPECTED_BATCH_WINDOWS,
            "effective_source_windows": EXPECTED_EFFECTIVE_WINDOWS,
            "parameter_updates_total_per_single_model_fold": EXPECTED_SOURCE_MINIBATCHES,
            "microbatch_windows": args.microbatch_windows,
            "gradient_accumulation_steps": EXPECTED_BATCH_WINDOWS // args.microbatch_windows,
        },
    })
    _write_json(output_root / "environment.json", _environment(device))
    cities, data_provenance = volume_protocol.load_cities()
    rows = _summary_rows(output_root)
    for target in EXPECTED_CITIES:
        for seed in EXPECTED_SEEDS:
            existing = next((row for row in rows if row.get("target_city") == target and row.get("seed") == seed), None)
            if args.resume and existing and existing.get("status") == "completed":
                print(f"skip completed target={target} seed={seed}", flush=True)
                continue
            print(f"start target={target} seed={seed} source_minibatches=2048 effective_windows=8192 parameter_updates=2048", flush=True)
            result = run_fold(
                cities, data_provenance, protocol,
                target=target, seed=seed, output_root=output_root,
                device_name=args.device, microbatch_windows=args.microbatch_windows,
                inference_batch_windows=args.inference_batch_windows,
                torch_threads=args.torch_threads, progress_every=args.progress_every,
            )
            rows = [row for row in rows if not (row.get("target_city") == target and row.get("seed") == seed)]
            rows.append(result)
            write_summary(output_root, protocol, rows, status="running")
            print(f"finish target={target} seed={seed} status={result.get('status')} metrics={result.get('metrics_raw_volume')}", flush=True)
    rows = _summary_rows(output_root)
    write_summary(output_root, protocol, rows, status="completed" if len(rows) == 18 and all(row.get("status") == "completed" for row in rows) else "completed_with_failures")
    return 0 if all(row.get("status") == "completed" for row in rows) else 2


def smoke() -> dict[str, Any]:
    protocol = load_frozen_protocol()
    _seed_everything(0)
    device = torch.device("cpu")
    model = MatchedVolumeTimeMixer().to(device).train()
    history = torch.randn(EXPECTED_BATCH_WINDOWS, EXPECTED_CONTEXT, 3)
    target = torch.randn(EXPECTED_BATCH_WINDOWS, EXPECTED_HORIZON, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad(set_to_none=True)
    for offset in range(EXPECTED_BATCH_WINDOWS):
        model_history, _ = _flatten_windows(history[offset:offset + 1], target[offset:offset + 1])
        prediction = model(model_history)
        prediction = prediction.reshape(1, 3, EXPECTED_HORIZON).transpose(1, 2)
        (_objective(prediction, target[offset:offset + 1]) / EXPECTED_BATCH_WINDOWS).backward()
    optimizer.step()
    return {
        "status": "passed",
        "protocol_file": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "input": "volume-only",
        "output_shape": [EXPECTED_BATCH_WINDOWS, EXPECTED_HORIZON, 3],
        "source_minibatches_total": EXPECTED_SOURCE_MINIBATCHES,
        "effective_source_windows": EXPECTED_EFFECTIVE_WINDOWS,
        "parameter_updates_total_per_single_model_fold": EXPECTED_SOURCE_MINIBATCHES,
        "physical_microbatch_windows": DEFAULT_MICROBATCH_WINDOWS,
        "gradient_accumulation_steps": EXPECTED_BATCH_WINDOWS,
        "september_read": False,
        "frozen_protocol_status": protocol.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--microbatch-windows", type=int, default=DEFAULT_MICROBATCH_WINDOWS)
    parser.add_argument("--inference-batch-windows", type=int, default=DEFAULT_INFERENCE_BATCH_WINDOWS)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(smoke(), indent=2, ensure_ascii=False))
        return 0
    if not args.run_all:
        parser.error("pass --smoke-only or --run-all")
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())

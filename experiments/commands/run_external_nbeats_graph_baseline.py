#!/usr/bin/env python3
"""Source-only N-BEATS-style behavior-graph transfer to three external panels.

Each seed trains one variable-site model on CHARGED source prefixes only,
saves the checkpoint, then evaluates Jiaxing, Shanghai, and UrbanEV. The local
implementation is a diagnostic baseline, not a scprc of a cited paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
OUT_ROOT = ROOT / "experiments/supplementary/external_strong_baselines_20261028/nbeats_graph"
CONTEXT, HORIZON, TRAIN_END = 168, 24, 2928
UPDATES, BATCH_WINDOWS = 2048, 4
SEEDS = (13, 17, 29)
CITY_CODES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiments.commands.run_external_dlinear_baselines as external_data  # noqa: E402
from scprc.training.nbeats_graph import (  # noqa: E402
    SharedStationNBEATSBehaviorGraphResidual,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def train(seed: int, device: torch.device, threads: int,
          progress_every: int) -> tuple[torch.nn.Module, dict[str, Any], float, float,
                                        dict[str, Any]]:
    cities, mean, std, provenance = external_data.load_charged_source()
    seed_all(seed)
    torch.set_num_threads(max(1, threads))
    model = SharedStationNBEATSBehaviorGraphResidual(
        hidden=128, blocks=4, graph_rank=16, temperature=0.15, residual_bound=0.20,
    ).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    starts = np.arange(CONTEXT, TRAIN_END - HORIZON + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    source_exposure = {city: 0 for city in CITY_CODES}
    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for minibatch in range(UPDATES):
        city = CITY_CODES[minibatch % len(CITY_CODES)]
        values = cities[city]["values"]
        picks = rng.choice(starts, size=BATCH_WINDOWS, replace=True)
        history = np.stack([values[start - CONTEXT:start] for start in picks]).astype(np.float32)
        target = np.stack([values[start:start + HORIZON] for start in picks]).astype(np.float32)
        history_tensor = torch.from_numpy((history - mean) / std).to(device)
        target_tensor = torch.from_numpy((target - mean) / std).to(device)
        prediction = model(history_tensor)
        loss = objective(prediction.float(), target_tensor.float())
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite N-BEATS graph loss at minibatch {minibatch + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        source_exposure[city] += BATCH_WINDOWS
        if progress_every and (minibatch + 1) % progress_every == 0:
            print(f"seed={seed} minibatch={minibatch + 1}/{UPDATES} city={city} "
                  f"loss={losses[-1]:.6f}", flush=True)
    training_seconds = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    details = {
        "source_minibatches_total": UPDATES,
        "source_minibatches_by_city": {
            city: source_exposure[city] // BATCH_WINDOWS for city in CITY_CODES
        },
        "effective_source_windows": UPDATES * BATCH_WINDOWS,
        "effective_source_windows_by_city": source_exposure,
        "parameter_updates_total": UPDATES,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "city_sampling": "round-robin over six CHARGED source cities",
        "mean_training_loss": float(np.mean(losses)),
        "training_seconds": training_seconds,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": peak_memory,
        "torch_threads": max(1, threads),
        "source_training_only": True,
        "target_data_opened_before_checkpoint": False,
    }
    return model, details, mean, std, provenance


def predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
            mean: float, std: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    predictions, truths = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_WINDOWS):
            selected = starts[offset:offset + BATCH_WINDOWS]
            history = np.stack([values[int(start) - CONTEXT:int(start)] for start in selected]).astype(np.float32)
            truth = np.stack([values[int(start):int(start) + HORIZON] for start in selected]).astype(np.float32)
            history_tensor = torch.from_numpy((history - mean) / std).to(device)
            forecast = model(history_tensor).cpu().numpy() * std + mean
            predictions.append(forecast.astype(np.float32))
            truths.append(truth)
    return np.concatenate(predictions), np.concatenate(truths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run one source-only seed")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        model = SharedStationNBEATSBehaviorGraphResidual(hidden=32, blocks=2)
        history = torch.randn(2, CONTEXT, 7)
        prediction = model(history)
        if prediction.shape != (2, HORIZON, 7) or not torch.isfinite(prediction).all():
            raise RuntimeError("N-BEATS graph smoke failed")
        print(json.dumps({"status": "passed", "output_shape": list(prediction.shape)}))
        return
    if not args.run or args.seed is None:
        parser.error("full training requires --run and --seed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_dir = OUT_ROOT / "checkpoints" / f"seed_{args.seed}"
    if checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model, training, mean, std, source_info = train(
        args.seed, device, args.threads, args.progress_every,
    )
    checkpoint = checkpoint_dir / "checkpoint.pt"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": args.seed, "source_scaler_mean": mean, "source_scaler_std": std}, checkpoint)
    code_paths = [Path(__file__).resolve(),
                  EXPERIMENTS / "nbeats_graph.py"]
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "cuda": torch.version.cuda,
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    write_json(checkpoint_dir / "training.json", {
        "status": "completed", "seed": args.seed, "training": training,
        "source_provenance": source_info, "source_scaler_mean": mean,
        "source_scaler_std": std, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "environment": environment,
        "protocol": {"context_hours": CONTEXT, "horizon_hours": HORIZON,
                     "target_variables": 1, "source_interval": "CHARGED 2023-04-01 through 2023-07-31",
                     "target_training_or_selection": False,
                     "retrospective_external_evaluation": True,
                     "model_scope": "local shared N-BEATS-style backbone + history-conditioned behavior-graph residual; not a paper scprc"},
    })
    for dataset in ("jiaxing", "shanghai", "urbanev"):
        dataset_dir = OUT_ROOT / dataset / f"seed_{args.seed}"
        if dataset_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=False)
        external = external_data.load_external(dataset)
        prediction, truth = predict(model, external["values"], external["starts"],
                                    mean, std, device)
        error = prediction.astype(np.float64) - truth.astype(np.float64)
        per_site = [{"site_id": str(external["site_ids"][index]),
                     "rmse": float(np.sqrt(np.mean(error[:, :, index] ** 2))),
                     "mae": float(np.mean(np.abs(error[:, :, index])))}
                    for index in range(error.shape[2])]
        metrics = {"pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
                   "pooled_mae": float(np.mean(np.abs(error))),
                   "n_origins": int(truth.shape[0]), "horizon": int(truth.shape[1]),
                   "n_sites": int(truth.shape[2]), "per_site": per_site}
        origin_times = np.asarray([str(external["times"][int(start)])
                                   for start in external["starts"]], dtype=str)
        prediction_file = dataset_dir / "predictions.npz"
        np.savez_compressed(prediction_file, truth=truth,
                            prediction_nbeats_graph=prediction,
                            forecast_starts=external["starts"],
                            forecast_origin_timestamps=origin_times,
                            site_ids=np.asarray(external["site_ids"], dtype=str),
                            unit=np.asarray(external["unit"]))
        external_metadata = {key: value for key, value in external.items()
                             if key not in {"values", "starts", "times", "site_ids"}}
        write_json(dataset_dir / "config.json", {
            "model": training["parameters"], "dataset": dataset, "seed": args.seed,
            "context_hours": CONTEXT, "horizon_hours": HORIZON,
            "source_checkpoint": str(checkpoint.relative_to(ROOT)),
            "target_training_or_tuning": False,
        })
        write_json(dataset_dir / "result.json", {
            "status": "completed", "model": "local source-only N-BEATS-style behavior-graph baseline",
            "dataset": dataset, "seed": args.seed, "unit": external["unit"],
            "metrics": metrics, "training": training,
            "source_provenance": source_info,
            "target_protocol": external_metadata,
            "information_boundary": {
                "target_data_opened_after_source_checkpoint_saved": True,
                "target_training_or_tuning": False,
                "test_score_model_selection": False,
                "retrospective_external_evaluation": True,
                "rolling_history_uses_observations_available_at_origin": True,
            },
            "artifacts": {"checkpoint": str(checkpoint.relative_to(ROOT)),
                          "predictions": str(prediction_file.relative_to(ROOT))},
            "environment": environment,
        })
        print(f"completed {dataset} seed={args.seed}: "
              f"RMSE={metrics['pooled_rmse']:.6f} MAE={metrics['pooled_mae']:.6f}", flush=True)


if __name__ == "__main__":
    main()

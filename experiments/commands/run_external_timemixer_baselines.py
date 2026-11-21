#!/usr/bin/env python3
"""Source-only TimeMixer transfer to Jiaxing, Shanghai, and UrbanEV.

The runner trains only on the frozen April--July CHARGED source prefixes.
Target arrays are opened only after each source-only checkpoint is written.
Outputs are new, dated artifacts and existing run directories are never
overwritten. ``--smoke`` uses synthetic tensors only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
CHARGED = ROOT / "scprc/data_adapters/charged"
PROTOCOL_PATH = EXPERIMENTS / 'scprc_matched_protocol.json'
EXPECTED_PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
OUT_ROOT = ROOT / "experiments/supplementary/external_strong_baselines_20261028/timemixer"
SEEDS = (13, 17, 29)
CITY_CODES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
CONTEXT, HORIZON, TRAIN_END = 168, 24, 2928

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as volume_protocol  # noqa: E402
import scprc.training.run_charged_timemixer_matched as timemixer_runner  # noqa: E402
import experiments.commands.run_external_dlinear_baselines as external_data  # noqa: E402
from scprc.data_adapters.charged.city_adapter import ARCHIVE, CITY_CODES as ARCHIVE_CITIES, FIRST_HOUR  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol hash mismatch: {digest}")
    protocol = json.loads(raw)
    if protocol["input"]["features"] != ["volume"]:
        raise ValueError("frozen protocol is not volume-only")
    budget = protocol["training_budget"]
    if (budget["source_minibatches_total"], budget["batch_size_windows"],
            budget["effective_source_windows"]) != (2048, 4, 8192):
        raise ValueError("frozen source exposure differs from expected protocol")
    return protocol


def load_source() -> tuple[dict[str, dict[str, Any]], float, float, dict[str, Any]]:
    if tuple(ARCHIVE_CITIES) != CITY_CODES:
        raise RuntimeError(f"unexpected CHARGED city order: {ARCHIVE_CITIES}")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    expected_times = pd.date_range(FIRST_HOUR, periods=TRAIN_END, freq="h")
    cities: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for city in CITY_CODES:
            with archive.open(f"{city}/volume.csv") as stream:
                frame = pd.read_csv(stream, nrows=TRAIN_END)
            times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
            site_ids = tuple(map(str, frame.columns[1:]))
            if len(frame) != TRAIN_END or not times.equals(expected_times):
                raise ValueError(f"invalid CHARGED April--July prefix for {city}")
            if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid source values for {city}")
            if not site_ids or len(set(site_ids)) != len(site_ids):
                raise ValueError(f"invalid site IDs for {city}")
            cities[city] = {"city_id": city, "times": times, "site_ids": site_ids,
                            "values": values}
    mean, std = volume_protocol.source_scaler(cities, CITY_CODES)
    provenance = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256(ARCHIVE),
        "source_cities": list(CITY_CODES),
        "rows_per_city": TRAIN_END,
        "source_interval": "2023-04-01 00:00 through 2023-07-31 23:00",
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "target_values_materialized_before_checkpoint": False,
    }
    return cities, mean, std, provenance


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
            mean: float, std: float, device: torch.device,
            batch_windows: int = 4) -> tuple[np.ndarray, np.ndarray]:
    predictions, truths = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_windows):
            selected = starts[offset:offset + batch_windows]
            history = np.stack([values[int(start) - CONTEXT:int(start)] for start in selected]).astype(np.float32)
            target = np.stack([values[int(start):int(start) + HORIZON] for start in selected]).astype(np.float32)
            scaled = (history - mean) / std
            windows, _, sites = scaled.shape
            model_input = torch.from_numpy(scaled).to(device)
            model_input = model_input.permute(0, 2, 1).contiguous().reshape(
                windows * sites, CONTEXT, 1)
            output = model(model_input).reshape(windows, sites, HORIZON).transpose(1, 2)
            prediction = output.cpu().numpy() * std + mean
            predictions.append(prediction.astype(np.float32))
            truths.append(target)
    return np.concatenate(predictions), np.concatenate(truths)


def metric_bundle(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    site_scores = []
    for site_index in range(error.shape[2]):
        site_error = error[:, :, site_index]
        site_scores.append({"rmse": float(np.sqrt(np.mean(site_error ** 2))),
                            "mae": float(np.mean(np.abs(site_error)))})
    return {
        "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
        "pooled_mae": float(np.mean(np.abs(error))),
        "equal_site_macro_rmse": float(np.mean([row["rmse"] for row in site_scores])),
        "equal_site_macro_mae": float(np.mean([row["mae"] for row in site_scores])),
        "aggregate_total_rmse": float(np.sqrt(np.mean(error.sum(axis=2) ** 2))),
        "aggregate_total_mae": float(np.mean(np.abs(error.sum(axis=2)))),
        "per_site": site_scores,
        "n_origins": int(error.shape[0]), "horizon": int(error.shape[1]),
        "n_sites": int(error.shape[2]),
    }


def fit_and_evaluate(dataset: str, seed: int, device_name: str,
                     torch_threads: int, progress_every: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    run_dir = OUT_ROOT / dataset / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {run_dir}")
    protocol = load_protocol()
    cities, mean, std, source_info = load_source()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "model": "official vendored TimeMixer channel-dependent volume-only route",
        "dataset": dataset, "seed": seed, "context_hours": CONTEXT,
        "horizon_hours": HORIZON, "source_cities": list(CITY_CODES),
        "target_training_or_tuning": False,
        "source_minibatches": 2048, "batch_windows": 4,
        "effective_source_windows": 8192, "optimizer_updates": 2048,
        "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "source-standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "city_sampling": "round-robin over all six CHARGED cities",
        "torch_threads": torch_threads, "device": device_name,
        "frozen_protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "frozen_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    seed_all(seed)
    started = time.perf_counter()
    model, training = timemixer_runner.fit_model(
        cities, CITY_CODES, mean, std, seed=seed,
        source_minibatches=2048, batch_windows=4, microbatch_windows=1,
        learning_rate=0.001, weight_decay=0.0001, device=device,
        torch_threads=torch_threads, progress_every=progress_every,
    )
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": seed, "dataset": dataset, "source_scaler_mean": mean,
                "source_scaler_std": std}, checkpoint)

    external = external_data.load_external(dataset)
    prediction, truth = predict(model, external["values"], external["starts"],
                                mean, std, device)
    if prediction.shape != truth.shape:
        raise RuntimeError(f"prediction/truth shape mismatch: {prediction.shape} != {truth.shape}")
    scores = metric_bundle(prediction, truth)
    scores["per_site"] = [
        {**row, "site_id": str(external["site_ids"][index])}
        for index, row in enumerate(scores["per_site"])
    ]
    prediction_file = run_dir / "predictions.npz"
    origin_times = np.asarray([str(external["times"][int(start)])
                               for start in external["starts"]], dtype=str)
    np.savez_compressed(prediction_file, truth=truth, prediction_timemixer=prediction,
                        forecast_starts=external["starts"],
                        forecast_origin_timestamps=origin_times,
                        site_ids=np.asarray(external["site_ids"], dtype=str),
                        unit=np.asarray(external["unit"]))
    external_metadata = {key: value for key, value in external.items()
                         if key not in {"values", "starts", "times", "site_ids"}}
    result = {
        "status": "completed", "model": "official vendored TimeMixer source-only transfer",
        "dataset": dataset, "seed": seed, "metrics": scores,
        "source_provenance": source_info, "source_scaler_mean": mean,
        "source_scaler_std": std, "training": training,
        "target_protocol": external_metadata,
        "information_boundary": {
            "source_training_and_scaler_only": True,
            "target_labels_read_after_source_checkpoint_saved": True,
            "target_training_or_tuning": False,
            "test_score_model_selection": False,
            "retrospective_external_evaluation": True,
            "rolling_history_may_use_observations_available_at_each_origin": True,
        },
        "artifacts": {"checkpoint": str(checkpoint.relative_to(ROOT)),
                      "predictions": str(prediction_file.relative_to(ROOT))},
        "elapsed_seconds_including_evaluation": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "cuda": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        "torch_threads": torch.get_num_threads()},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(),
            EXPERIMENTS / "run_charged_timemixer_matched.py",
            EXPERIMENTS / "training_protocol.py",
        )},
    }
    (run_dir / "result.json").write_text(json.dumps(
        result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"completed {dataset} seed={seed}: RMSE={scores['pooled_rmse']:.6f} MAE={scores['pooled_mae']:.6f}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir / "result.json"


def smoke_test() -> None:
    protocol = load_protocol()
    torch.manual_seed(0)
    model = timemixer_runner.MatchedVolumeTimeMixer().cpu().train()
    history = torch.randn(4, CONTEXT, 3)
    target = torch.randn(4, HORIZON, 3)
    model_input = history.permute(0, 2, 1).contiguous().reshape(12, CONTEXT, 1)
    output = model(model_input).reshape(4, 3, HORIZON).transpose(1, 2)
    loss = timemixer_runner._objective(output, target)
    loss.backward()
    if output.shape != target.shape or not torch.isfinite(loss):
        raise AssertionError("TimeMixer synthetic smoke failed")
    print(json.dumps({"status": "passed", "training_started": False,
                      "output_shape": list(output.shape), "loss": float(loss.detach()),
                      "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                      "source_minibatches": 2048, "effective_windows": 8192},
                     indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--dataset", choices=("jiaxing", "shanghai", "urbanev", "all"), default="all")
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=256)
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return 0
    protocol = load_protocol()
    datasets = ("jiaxing", "shanghai", "urbanev") if args.dataset == "all" else (args.dataset,)
    seeds = tuple(args.seed) if args.seed else SEEDS
    destinations = [OUT_ROOT / dataset / f"seed_{seed}" for dataset in datasets for seed in seeds]
    collisions = [path for path in destinations if path.exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, collisions)))
    for dataset in datasets:
        for seed in seeds:
            fit_and_evaluate(dataset, seed, args.device, args.torch_threads, args.progress_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

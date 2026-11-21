#!/usr/bin/env python3
"""Matched source-only PatchTST transfer runs for three external datasets.

This runner never fits, scales, selects, or adapts on external target labels.
All three datasets train on the six frozen CHARGED source prefixes only.
Generated files are written below a new dated paper2 supplement directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
CHARGED = ROOT / "scprc/data_adapters/charged"
PROTOCOL_PATH = EXPERIMENTS / 'scprc_matched_protocol.json'
PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
OUT_ROOT = ROOT / "experiments/supplementary/external_strong_baselines_20261028/patchtst"
JIA_BASE = EXPERIMENTS / "jiaxing_scpcr_external_20261021"
URBAN_CSV = ROOT / "research/urbanev_timemixer/sources/UrbanEV/data/volume.csv"
SHANG_BASE = ROOT / "experiments/supplementary/external_current_20261024/shanghai"
SEEDS = (13, 17, 29)
CONTEXT, HORIZON, TRAIN_END = 168, 24, 2928

for import_path in (ROOT,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import scprc.training.training_protocol as charged_protocol  # noqa: E402
import scprc.training.run_charged_patchtst_volume_only_loco as patch_runner  # noqa: E402
import experiments.commands.run_external_dlinear_baselines as external_data  # noqa: E402
from scprc.data_adapters.charged.city_adapter import ARCHIVE, CITY_CODES, FIRST_HOUR  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protocol() -> dict[str, Any]:
    data = PROTOCOL_PATH.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PROTOCOL_SHA256:
        raise RuntimeError(f"frozen CHARGED protocol hash mismatch: {actual}")
    frozen = json.loads(data)
    if frozen["input"]["context_hours"] != CONTEXT or frozen["input"]["forecast_horizon_hours"] != HORIZON:
        raise RuntimeError("frozen protocol is not 168/24")
    budget = frozen["training_budget"]
    expected = (2048, 4, 8192, 0.001, 0.0001)
    observed = (budget["source_minibatches_total"], budget["batch_size_windows"],
                budget["effective_source_windows"], budget["learning_rate"], budget["weight_decay"])
    if observed != expected:
        raise RuntimeError(f"unexpected frozen training budget: {observed}")
    return frozen


def load_source_train() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read only CHARGED Apr-Jul rows; August and target data are not materialized."""
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    cities: dict[str, dict[str, Any]] = {}
    import zipfile

    with zipfile.ZipFile(ARCHIVE) as archive:
        for city in CITY_CODES:
            member = f"{city}/volume.csv"
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, nrows=TRAIN_END)
            if frame.shape[0] != TRAIN_END or frame.shape[1] < 2:
                raise ValueError(f"invalid source prefix shape for {member}: {frame.shape}")
            times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            expected_times = pd.date_range(FIRST_HOUR, periods=TRAIN_END, freq="h")
            if not times.equals(expected_times):
                raise ValueError(f"unexpected source timestamps in {member}")
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
            site_ids = tuple(map(str, frame.columns[1:]))
            if not site_ids or len(set(site_ids)) != len(site_ids):
                raise ValueError(f"invalid site IDs in {member}")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid load values in {member}")
            cities[city] = {"city_id": city, "times": times, "site_ids": site_ids, "values": values}
    return cities, {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256(ARCHIVE),
        "source_cities": list(CITY_CODES),
        "source_rows_per_city": TRAIN_END,
        "source_interval": "2023-04-01 00:00 through 2023-07-31 23:00",
        "august_labels_materialized": False,
    }


def _load_jiaxing_after_fit() -> dict[str, Any]:
    freeze_path = JIA_BASE / "protocol_freeze.json"
    panel_path = JIA_BASE / "jiaxing_hourly_panel.npz"
    if not freeze_path.is_file() or not panel_path.is_file():
        raise FileNotFoundError(f"Jiaxing frozen panel missing: {panel_path}")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("prepared_sha256") != sha256(panel_path):
        raise ValueError("Jiaxing prepared panel does not match its frozen hash")
    with np.load(panel_path, allow_pickle=False) as archive:
        panel = {key: archive[key] for key in archive.files}
    values = panel["energy_kwh"].astype(np.float32)
    starts = panel["test_starts"].astype(np.int64)
    sites = panel["site_ids"].astype(str)
    if values.ndim != 2 or values.shape[1] != len(sites) or values.shape[1] != 13:
        raise ValueError(f"unexpected Jiaxing panel dimensions: {values.shape}")
    if len(starts) != 2185 or starts[0] != int(panel["target_prefix_end"]):
        raise ValueError("Jiaxing locked test origins mismatch")
    if starts[-1] + HORIZON != int(panel["test_end"]):
        raise ValueError("Jiaxing test boundary mismatch")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid Jiaxing energy values")
    return {"values": values, "starts": starts, "site_ids": sites,
            "timestamps": panel["times"].astype(str), "unit": "kWh per station-hour",
            "dataset_sha256": sha256(panel_path), "freeze": frozen}


def _load_urbanev_after_fit() -> dict[str, Any]:
    if not URBAN_CSV.is_file():
        raise FileNotFoundError(URBAN_CSV)
    frame = pd.read_csv(URBAN_CSV)
    if frame.shape != (4344, 276):
        raise ValueError(f"unexpected UrbanEV volume.csv shape: {frame.shape}")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range("2022-09-01 00:00:00", periods=4344, freq="h")
    if not times.equals(expected):
        raise ValueError("UrbanEV timestamps do not match the locked hourly panel")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    site_ids = np.asarray(frame.columns[1:], dtype=str)
    starts = np.arange(3672, len(values) - HORIZON + 1, dtype=np.int64)
    if len(starts) != 649 or starts[0] != 3672:
        raise ValueError("UrbanEV Feb-2023 test origin mismatch")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid UrbanEV hourly volume values")
    return {"values": values, "starts": starts, "site_ids": site_ids,
            "timestamps": np.asarray(times.astype(str)), "unit": "volume.csv native hourly unit",
            "dataset_sha256": sha256(URBAN_CSV)}


def predict_external(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
                     mean: float, std: float, device: str, batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    predictions, truths = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            x = np.stack([values[start - CONTEXT:start] for start in batch_starts]).astype(np.float32)
            y = np.stack([values[start:start + HORIZON] for start in batch_starts]).astype(np.float32)
            x = (x - mean) / std
            batch, steps, sites = x.shape
            model_x = torch.from_numpy(x).permute(0, 2, 1).contiguous().reshape(
                batch * sites, steps, 1).to(device)
            pred_scaled = model(model_x).reshape(batch, sites, HORIZON).transpose(1, 2)
            pred = pred_scaled.cpu().numpy() * std + mean
            predictions.append(pred.astype(np.float32))
            truths.append(y)
    return np.concatenate(predictions), np.concatenate(truths)


def metric_bundle(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    per_site = []
    for index in range(error.shape[2]):
        site_error = error[:, :, index]
        per_site.append({"rmse": float(np.sqrt(np.mean(site_error ** 2))),
                         "mae": float(np.mean(np.abs(site_error)))})
    return {
        "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
        "pooled_mae": float(np.mean(np.abs(error))),
        "macro_site_rmse": float(np.mean([item["rmse"] for item in per_site])),
        "macro_site_mae": float(np.mean([item["mae"] for item in per_site])),
        "per_site": per_site,
    }


def run_external(dataset: str, seed: int, device: str, torch_threads: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    out_dir = OUT_ROOT / dataset / f"seed_{seed}"
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_dir}")
    assert_protocol()
    cities, source_info = load_source_train()
    source_ids = tuple(CITY_CODES)
    mean, std = charged_protocol.source_scaler(cities, source_ids)
    if dataset not in {"jiaxing", "shanghai", "urbanev"}:
        raise ValueError(f"unsupported external dataset: {dataset}")

    model, training = patch_runner._fit_model(
        cities, source_ids, mean, std, seed=seed, device=device, torch_threads=torch_threads)
    if dataset == "jiaxing":
        target = _load_jiaxing_after_fit()
    elif dataset == "urbanev":
        target = _load_urbanev_after_fit()
    else:
        target = external_data.load_external("shanghai")
        target = {
            "values": target["values"], "starts": target["starts"],
            "site_ids": target["site_ids"], "timestamps": target["times"],
            "unit": target["unit"], "dataset_sha256": target["data_sha256"],
            "protocol": target,
        }
    prediction, truth = predict_external(model, target["values"], target["starts"],
                                         mean, std, device)
    out_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = out_dir / "predictions.npz"
    timestamps = target["timestamps"]
    origin_timestamps = np.asarray([timestamps[int(start)] for start in target["starts"]], dtype=str)
    np.savez_compressed(prediction_path, label_starts=target["starts"],
                        origin_timestamps=origin_timestamps, site_ids=target["site_ids"],
                        truth=truth, prediction_patchtst=prediction)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint_path)
    result = {
        "status": "completed", "dataset": dataset, "seed": seed,
        "model": "pinned official univariate PatchTST, channel-independent",
        "metrics": metric_bundle(prediction, truth), "origins": len(target["starts"]),
        "site_count": len(target["site_ids"]), "unit": target["unit"],
        "target_dataset_sha256": target["dataset_sha256"],
        "prediction_file": str(prediction_path), "checkpoint_file": str(checkpoint_path),
        "source_provenance": source_info, "source_scaler_mean": mean,
        "source_scaler_std": std, "training": training,
        "target_protocol": ({key: value for key, value in target["protocol"].items()
                              if key not in {"values", "starts", "times", "site_ids"}}
                             if isinstance(target.get("protocol"), dict) else target.get("protocol")),
        "protocol": {
            "frozen_protocol": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "frozen_protocol_sha256": PROTOCOL_SHA256,
            "context_hours": CONTEXT, "horizon_hours": HORIZON,
            "source_only_training_and_scaler": True, "target_adaptation": False,
            "target_test_label_model_selection": False,
            "source_minibatches": 2048, "batch_size_windows": 4,
            "effective_source_windows": 8192,
            "source_schedule": "round-robin PCGrad over six CHARGED cities",
            "parameter_updates": training["parameter_updates"],
            "city_minibatches": training["source_minibatches_by_city"],
            "rolling_observed_history": True,
        },
        "code": {"runner_sha256": sha256(HERE),
                 "patchtst_runner_sha256": sha256(EXPERIMENTS / "run_charged_patchtst_volume_only_loco.py"),
                 "patchtst_adapter_sha256": sha256(ROOT / "scprc/baselines/patchtst_adapter.py"),
                 "external_data_loader_sha256": sha256(ROOT / "experiments/commands/run_external_dlinear_baselines.py")},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "device": device, "cuda_available": torch.cuda.is_available()},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                          encoding="utf-8")
    del model
    return out_dir / "result.json"


def audit_shanghai(seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    out_dir = OUT_ROOT / "shanghai" / f"seed_{seed}"
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_dir}")
    run_dir = SHANG_BASE / f"seed{seed}"
    freeze_path, result_path = run_dir / "freeze.json", run_dir / "result.json"
    pred_path, checkpoint_path = run_dir / "q4_predictions.npz", run_dir / "patchtst.pt"
    history_path = SHANG_BASE / "mp_evdata_history_prefix.npz"
    protocol_path = SHANG_BASE / "protocol_freeze.json"
    required = (freeze_path, result_path, pred_path, checkpoint_path, history_path, protocol_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Shanghai reuse artifacts: " + ", ".join(missing))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if int(frozen.get("seed", -1)) != seed or int(result.get("seed", -1)) != seed:
        raise ValueError("Shanghai seed metadata mismatch")
    if protocol.get("context_hours") != CONTEXT or protocol.get("horizon_hours") != HORIZON:
        raise ValueError("Shanghai protocol is not 168/24")
    if protocol.get("target_training_or_scaling") is not False:
        raise ValueError("Shanghai metadata does not certify source-only target handling")
    patch_meta = frozen.get("training", {}).get("patchtst", {})
    if (patch_meta.get("batch_size_windows") != 4 or patch_meta.get("steps") != 2048 or
            patch_meta.get("optimizer") != "AdamW"):
        raise ValueError("Shanghai PatchTST training metadata is incomplete or incompatible")
    with np.load(history_path, allow_pickle=False) as history_archive:
        history = {key: history_archive[key] for key in history_archive.files}
    with np.load(pred_path, allow_pickle=False) as prediction_archive:
        arrays = {key: prediction_archive[key] for key in prediction_archive.files}
    sites = tuple(map(str, protocol["site_ids"]))
    if tuple(map(str, history["site_ids"])) != sites or tuple(map(str, arrays["site_ids"])) != sites:
        raise ValueError("Shanghai site ordering differs among protocol, history, and predictions")
    if history["history_kw"].shape != (6576, len(sites)):
        raise ValueError(f"unexpected Shanghai history shape {history['history_kw'].shape}")
    starts = arrays["label_starts"]
    if starts.shape != (2185,) or starts[0] != 6576 or starts[-1] != 8760 or not np.all(np.diff(starts) == 1):
        raise ValueError("Shanghai Q4 forecast-origin sequence mismatch")
    if arrays["truth_kw"].shape != (2185, HORIZON, len(sites)):
        raise ValueError("Shanghai truth tensor has unexpected shape")
    if arrays["pred_patchtst_kw"].shape != arrays["truth_kw"].shape:
        raise ValueError("Shanghai PatchTST prediction shape mismatch")
    if not np.isfinite(arrays["truth_kw"]).all() or not np.isfinite(arrays["pred_patchtst_kw"]).all():
        raise ValueError("Shanghai prediction cache contains nonfinite values")
    saved_updates = patch_meta.get("updates_by_city", {})
    optimizer_update_count = int(sum(saved_updates.values())) if saved_updates else None
    audit = {
        "status": "reused_and_audited_no_retraining", "dataset": "Shanghai MP-EVData",
        "seed": seed, "existing_result": str(result_path), "existing_checkpoint": str(checkpoint_path),
        "existing_prediction_cache": str(pred_path), "existing_result_sha256": sha256(result_path),
        "existing_checkpoint_sha256": sha256(checkpoint_path), "prediction_cache_sha256": sha256(pred_path),
        "origins": int(len(starts)), "sites": list(sites), "history_shape": list(history["history_kw"].shape),
        "truth_shape": list(arrays["truth_kw"].shape), "patchtst_prediction_shape": list(arrays["pred_patchtst_kw"].shape),
        "test_interval": protocol["external_test_interval"],
        "target_training_or_scaling": protocol["target_training_or_scaling"],
        "existing_training": patch_meta,
        "existing_optimizer_updates_total": optimizer_update_count,
        "matched_budget_note": "Existing Shanghai PatchTST reuses 2048 source minibatches x batch 4, but its archived runner uses 2048 single-task AdamW updates, not the six-source PCGrad update schedule (342 updates) used by this new Jiaxing/UrbanEV runner; report this lineage difference.",
        "metrics_reused_from_result": result.get("metrics", {}).get("patchtst"),
        "training_performed": False,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    audit_path = out_dir / "reuse_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
    return audit_path


def smoke_test() -> dict[str, Any]:
    frozen = assert_protocol()
    smoke = patch_runner.cpu_smoke()
    batch_counts = [2048 // len(CITY_CODES) + (index < 2048 % len(CITY_CODES))
                    for index in range(len(CITY_CODES))]
    if sum(batch_counts) != 2048 or max(batch_counts) != 342 or min(batch_counts) != 341:
        raise AssertionError("six-city source schedule does not allocate 2048 minibatches")
    return {"status": "passed", "training_started": False,
            "protocol_id": frozen["protocol_id"], "protocol_sha256": PROTOCOL_SHA256,
            "synthetic_model_smoke": smoke, "six_source_minibatches": batch_counts,
            "expected_pcgrad_parameter_updates": 342,
            "output_created": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("jiaxing", "shanghai", "urbanev"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke-only", action="store_true", help="synthetic forward/gradient test only; no training or output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.smoke_only:
        if args.dataset or args.seed:
            parser.error("--smoke-only cannot be combined with --dataset or --seed")
        print(json.dumps(smoke_test(), indent=2, ensure_ascii=False))
        return 0
    if not args.dataset or args.seed is None:
        parser.error("--dataset and --seed are required unless --smoke-only is used")
    path = run_external(args.dataset, args.seed, args.device, args.torch_threads)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

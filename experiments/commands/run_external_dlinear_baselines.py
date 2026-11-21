#!/usr/bin/env python3
"""Source-only CHARGED DLinear baselines for three frozen external datasets.

Full runs require the explicit ``--run`` flag. ``--smoke`` uses synthetic data
and a temporary directory; it never reads the research datasets or writes to
the experiment output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT_ROOT = ROOT / "experiments/supplementary/external_strong_baselines_20261028/dlinear"
EXPERIMENTS = ROOT / 'scprc/training'
CHARGED_ARCHIVE = ROOT / "scprc/data_adapters/charged/Hourly.zip"
JIA_BASE = EXPERIMENTS / "jiaxing_scpcr_external_20261021"
SHANGHAI_BASE = EXPERIMENTS / "mp_evdata_scpcr_external_20261021"
SHANGHAI_FREEZE = ROOT / "experiments/supplementary/external_current_20261024/shanghai/protocol_freeze.json"
SHANGHAI_RAW = ROOT / "scprc/data_adapters/public/mp_evdata_figshare_29882366/station-level load Profile 1h.xlsx"
URBAN_RAW = ROOT / "research/urbanev_timemixer/sources/UrbanEV/data/volume.csv"
URBAN_PROTOCOL = EXPERIMENTS / "run_urbanev_external_scprc.py"

CONTEXT = 168
HORIZON = 24
TRAIN_END = 2928
TRAIN_UPDATES = 2048
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
SEEDS = (13, 17, 29)
CITY_CODES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SHANGHAI_SITES = ("A1", "A2", "A3", "A4", "A5", "A6", "A9")
DATASETS = ("jiaxing", "shanghai", "urbanev")

for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from scprc.models.linear import DLinear  # noqa: E402
from scprc.data_adapters.charged.city_adapter import ARCHIVE_MD5, FIRST_HOUR  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "git_commit": commit,
        "torch_num_threads": torch.get_num_threads(),
    }


def source_config() -> dict[str, Any]:
    return {
        "source_dataset": "CHARGED v3",
        "source_cities": list(CITY_CODES),
        "source_interval": "2023-04-01 00:00 through 2023-07-31 23:00, local published hourly grid",
        "source_rows_per_city": TRAIN_END,
        "context_hours": CONTEXT,
        "horizon_hours": HORIZON,
        "target_variables": 1,
        "batch_size_windows": BATCH_SIZE,
        "optimizer_updates": TRAIN_UPDATES,
        "effective_window_exposures": TRAIN_UPDATES * BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "MSE on source-standardized univariate site load",
        "city_sampling": "round-robin city per update; sample four train windows with replacement",
        "scaler": "equal-city-weighted mean and second moment over source rows [0,2928); target data excluded",
        "model": "LTSF-Linear DLinear, moving average 25, individual=False, channels=1",
        "architecture": "endpoint-replicated centered moving-average decomposition; separate shared seasonal/trend Linear(168,24) projections; sum branches; each site is an independent sample",
        "source_reference_commit": "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
        "budget_basis": "matches the local CHARGED linear runner's 2048-update option, batch 4 and Adam lr=1e-3; this external transfer uses all six source cities, not an internal LOCO fold",
        "retrospective_evaluation": True,
    }


def load_charged_source() -> tuple[dict[str, dict[str, Any]], float, float, dict[str, Any]]:
    import zipfile

    if not CHARGED_ARCHIVE.is_file():
        raise FileNotFoundError(CHARGED_ARCHIVE)
    archive_md5 = file_md5(CHARGED_ARCHIVE)
    if archive_md5 != ARCHIVE_MD5:
        raise ValueError(f"CHARGED archive MD5 mismatch: {archive_md5}")
    cities: dict[str, dict[str, Any]] = {}
    expected = pd.date_range(FIRST_HOUR, periods=TRAIN_END, freq="h")
    with zipfile.ZipFile(CHARGED_ARCHIVE) as archive:
        for city in CITY_CODES:
            member = f"{city}/volume.csv"
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, nrows=TRAIN_END)
            times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
            site_ids = tuple(str(value) for value in frame.columns[1:])
            if len(frame) != TRAIN_END or not times.equals(expected):
                raise ValueError(f"unexpected frozen CHARGED source prefix: {member}")
            if not site_ids or len(set(site_ids)) != len(site_ids):
                raise ValueError(f"invalid site IDs in {member}")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid load values in {member}")
            cities[city] = {"values": values, "site_ids": site_ids}

    city_means = [float(cities[city]["values"].astype(np.float64).mean()) for city in CITY_CODES]
    city_seconds = [float(np.square(cities[city]["values"].astype(np.float64)).mean()) for city in CITY_CODES]
    mean = float(np.mean(city_means))
    std = max(float(np.sqrt(max(float(np.mean(city_seconds)) - mean * mean, 0.0))), 1e-6)
    provenance = {
        "archive": str(CHARGED_ARCHIVE.relative_to(ROOT)),
        "archive_md5": archive_md5,
        "archive_sha256": sha256_file(CHARGED_ARCHIVE),
        "source_prefix_sha256": {
            city: hashlib.sha256(
                cities[city]["values"].tobytes() + "\n".join(cities[city]["site_ids"]).encode()
            ).hexdigest()
            for city in CITY_CODES
        },
        "source_site_counts": {city: len(cities[city]["site_ids"]) for city in CITY_CODES},
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "rows_used_per_city": TRAIN_END,
    }
    return cities, mean, std, provenance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_dlinear(cities: dict[str, dict[str, Any]], mean: float, std: float,
                seed: int, device: torch.device, updates: int = TRAIN_UPDATES
                ) -> tuple[DLinear, torch.optim.Optimizer, float]:
    seed_everything(seed)
    model = DLinear(CONTEXT, HORIZON, channels=1, individual=False, moving_avg=25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts = np.arange(CONTEXT, TRAIN_END - HORIZON + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    loss_sum = 0.0
    model.train()
    for update in range(updates):
        city = CITY_CODES[update % len(CITY_CODES)]
        values = cities[city]["values"]
        picks = rng.choice(starts, size=BATCH_SIZE, replace=True)
        x = np.stack([values[s - CONTEXT:s] for s in picks]).astype(np.float32)
        y = np.stack([values[s:s + HORIZON] for s in picks]).astype(np.float32)
        batch, _, nodes = x.shape
        x_tensor = torch.from_numpy((x - mean) / std).permute(0, 2, 1).contiguous()
        x_tensor = x_tensor.reshape(batch * nodes, CONTEXT, 1).to(device)
        y_tensor = torch.from_numpy((y - mean) / std).transpose(1, 2).contiguous()
        y_tensor = y_tensor.reshape(batch * nodes, HORIZON, 1).to(device)
        prediction = model(x_tensor)
        loss = F.mse_loss(prediction, y_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
    return model, optimizer, loss_sum / updates


def load_external(dataset: str) -> dict[str, Any]:
    """Load frozen target values only after the caller has saved the checkpoint."""
    if dataset == "jiaxing":
        freeze_path = JIA_BASE / "protocol_freeze.json"
        panel_path = JIA_BASE / "jiaxing_hourly_panel.npz"
        freeze = json.loads(freeze_path.read_text())
        if sha256_file(panel_path) != freeze["prepared_sha256"]:
            raise ValueError("Jiaxing prepared panel hash differs from frozen protocol")
        with np.load(panel_path, allow_pickle=False) as panel:
            values = panel["energy_kwh"].astype(np.float32)
            starts = panel["test_starts"].astype(np.int64)
            times = panel["times"].astype(str)
            site_ids = panel["site_ids"].astype(str)
        if values.shape != (15336, 13) or starts.size != 2185:
            raise ValueError("Jiaxing panel/origin shape differs from frozen protocol")
        return {"values": values, "starts": starts, "times": times,
                "site_ids": site_ids, "unit": "kWh per location per hour",
                "protocol_path": str(freeze_path.relative_to(ROOT)),
                "protocol_sha256": sha256_file(freeze_path),
                "data_path": str(panel_path.relative_to(ROOT)),
                "data_sha256": sha256_file(panel_path),
                "split": "2021-07-01 through 2021-09-30; 2185 rolling origins"}

    if dataset == "shanghai":
        freeze = json.loads(SHANGHAI_FREEZE.read_text())
        if sha256_file(SHANGHAI_RAW) != freeze["raw_sha256"]:
            raise ValueError("Shanghai raw workbook hash differs from frozen protocol")
        prefix_path = SHANGHAI_BASE / "mp_evdata_history_prefix.npz"
        if sha256_file(prefix_path) != freeze["prepared_history_sha256"]:
            raise ValueError("Shanghai prepared prefix hash differs from frozen protocol")
        with np.load(prefix_path, allow_pickle=False) as panel:
            prefix = panel["history_kw"].astype(np.float32)
            site_ids = panel["site_ids"].astype(str)
            prefix_times = panel["history_timestamps"].astype(str)
        if tuple(site_ids.tolist()) != SHANGHAI_SITES or prefix.shape != (6576, 7):
            raise ValueError("Shanghai frozen prefix schema mismatch")
        expected_times = pd.date_range("2024-01-01 00:00", periods=8784, freq="h")
        test_by_site = []
        for site in SHANGHAI_SITES:
            frame = pd.read_excel(SHANGHAI_RAW, sheet_name=site, nrows=8784)
            times = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="coerce"))
            if not times.equals(expected_times):
                raise ValueError(f"Shanghai frozen timestamp check failed: {site}")
            vals = pd.to_numeric(frame["power"], errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(vals).all() or (vals < 0).any():
                raise ValueError(f"invalid Shanghai series: {site}")
            test_by_site.append(vals[6576:8784])
        test = np.stack(test_by_site, axis=1)
        values = np.concatenate((prefix, test), axis=0)
        starts = np.arange(6576, 8784 - HORIZON + 1, dtype=np.int64)
        return {"values": values, "starts": starts,
                "times": np.asarray(expected_times.astype(str)), "site_ids": site_ids,
                "unit": "kW", "protocol_path": str(SHANGHAI_FREEZE.relative_to(ROOT)),
                "protocol_sha256": sha256_file(SHANGHAI_FREEZE),
                "data_path": str(SHANGHAI_RAW.relative_to(ROOT)),
                "data_sha256": sha256_file(SHANGHAI_RAW),
                "split": "2024-10-01 through 2024-12-31; 2185 rolling origins",
                "target_labels_opened_after_checkpoint": True,
                "history_prefix_sha256": sha256_file(prefix_path),
                "history_timestamps_count": len(prefix_times)}

    if dataset == "urbanev":
        frame = pd.read_csv(URBAN_RAW)
        if frame.shape != (4344, 276):
            raise ValueError(f"UrbanEV expected 4344x276, observed {frame.shape}")
        times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
        expected = pd.date_range("2022-09-01 00:00", periods=4344, freq="h")
        if not times.equals(expected):
            raise ValueError("UrbanEV timestamp grid differs from frozen runner")
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        site_ids = np.asarray([str(x) for x in frame.columns[1:]])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid UrbanEV target values")
        starts = np.arange(3672, 4344 - HORIZON + 1, dtype=np.int64)
        return {"values": values, "starts": starts,
                "times": np.asarray(times.astype(str)), "site_ids": site_ids,
                "unit": "UrbanEV published volume unit", 
                "protocol_path": str(URBAN_PROTOCOL.relative_to(ROOT)),
                "protocol_sha256": sha256_file(URBAN_PROTOCOL),
                "data_path": str(URBAN_RAW.relative_to(ROOT)),
                "data_sha256": sha256_file(URBAN_RAW),
                "split": "2023-02-01 through 2023-02-28; 649 rolling origins"}
    raise ValueError(dataset)


def predict(model: DLinear, values: np.ndarray, starts: np.ndarray,
            mean: float, std: float, device: torch.device,
            batch_size: int = 16) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            selected = starts[offset:offset + batch_size]
            x = np.stack([values[s - CONTEXT:s] for s in selected]).astype(np.float32)
            batch, _, nodes = x.shape
            tensor = torch.from_numpy((x - mean) / std).permute(0, 2, 1).contiguous()
            tensor = tensor.reshape(batch * nodes, CONTEXT, 1).to(device)
            forecast = model(tensor).reshape(batch, nodes, HORIZON).transpose(1, 2)
            chunks.append(forecast.detach().cpu().numpy() * std + mean)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def summarize(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    per_site = []
    for index in range(truth.shape[2]):
        site_error = error[:, :, index]
        per_site.append({
            "site_index": index,
            "rmse": float(np.sqrt(np.mean(site_error ** 2))),
            "mae": float(np.mean(np.abs(site_error))),
        })
    total_error = error.sum(axis=2)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
        "pooled_mae": float(np.mean(np.abs(error))),
        "equal_site_macro_rmse": float(np.mean([row["rmse"] for row in per_site])),
        "equal_site_macro_mae": float(np.mean([row["mae"] for row in per_site])),
        "aggregate_total_rmse": float(np.sqrt(np.mean(total_error ** 2))),
        "aggregate_total_mae": float(np.mean(np.abs(total_error))),
        "per_site": per_site,
        "n_origins": int(truth.shape[0]),
        "horizon": int(truth.shape[1]),
        "n_sites": int(truth.shape[2]),
    }


def run_one(dataset: str, seed: int, source: dict[str, dict[str, Any]],
            mean: float, std: float, source_provenance: dict[str, Any],
            device: torch.device) -> None:
    run_dir = OUT_ROOT / dataset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **source_config(),
        "dataset": dataset,
        "seed": seed,
        "target_scaler": "none; source CHARGED scaler only",
        "target_training_or_tuning": False,
        "run_directory": str(run_dir.relative_to(ROOT)),
    }
    json_write(run_dir / "config.json", config)
    json_write(run_dir / "environment.json", environment())

    model, optimizer, mean_loss = fit_dlinear(source, mean, std, seed, device)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed,
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "source_provenance": source_provenance,
        "config": config,
        "mean_training_loss": mean_loss,
    }, checkpoint_path)

    # External labels are opened only after the source-only checkpoint is saved.
    external = load_external(dataset)
    prediction = predict(model, external["values"], external["starts"], mean, std, device)
    truth = np.stack([
        external["values"][int(start):int(start) + HORIZON]
        for start in external["starts"]
    ]).astype(np.float32)
    if prediction.shape != truth.shape:
        raise ValueError(f"forecast/truth mismatch: {prediction.shape} vs {truth.shape}")
    metrics = summarize(truth, prediction)
    metrics["per_site"] = [
        {**row, "site_id": str(external["site_ids"][row["site_index"]])}
        for row in metrics["per_site"]
    ]
    origin_times = [str(external["times"][int(start)]) for start in external["starts"]]
    np.savez_compressed(
        run_dir / "predictions.npz",
        truth=truth,
        prediction=prediction,
        forecast_starts=external["starts"],
        forecast_origin_timestamps=np.asarray(origin_times, dtype=str),
        site_ids=np.asarray(external["site_ids"], dtype=str),
        unit=np.asarray(external["unit"]),
    )
    result = {
        "status": "completed",
        "model": "DLinear univariate source-only transfer",
        "seed": seed,
        "dataset": dataset,
        "unit": external["unit"],
        "split": external["split"],
        "source_training": {**source_provenance, "mean_training_loss": mean_loss,
                            "optimizer_updates": TRAIN_UPDATES,
                            "batch_size_windows": BATCH_SIZE},
        "target_protocol": {k: v for k, v in external.items()
                            if k not in {"values", "starts", "times", "site_ids"}},
        "metrics": metrics,
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "predictions": str((run_dir / "predictions.npz").relative_to(ROOT)),
        },
    }
    json_write(run_dir / "result.json", result)
    print(f"completed {dataset} seed={seed}: RMSE={metrics['pooled_rmse']:.6f}, MAE={metrics['pooled_mae']:.6f}")


def smoke_test() -> None:
    device = torch.device("cpu")
    rng = np.random.default_rng(28)
    cities = {
        city: {"values": rng.normal(size=(TRAIN_END, 3 + i % 2)).astype(np.float32)}
        for i, city in enumerate(CITY_CODES)
    }
    mean, std = 0.0, 1.0
    model, optimizer, train_loss = fit_dlinear(cities, mean, std, 13, device, updates=2)
    toy_values = rng.normal(size=(CONTEXT + HORIZON + 2, 2)).astype(np.float32)
    starts = np.asarray([CONTEXT, CONTEXT + 1], dtype=np.int64)
    forecast = predict(model, toy_values, starts, mean, std, device, batch_size=2)
    truth = np.stack([toy_values[s:s + HORIZON] for s in starts]).astype(np.float32)
    metrics = summarize(truth, forecast)
    with tempfile.TemporaryDirectory(prefix="dlinear_smoke_") as temporary:
        output = Path(temporary)
        json_write(output / "config.json", {**source_config(), "smoke": True, "updates": 2})
        json_write(output / "environment.json", environment())
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, output / "checkpoint.pt")
        np.savez_compressed(output / "predictions.npz", truth=truth, prediction=forecast,
                            forecast_starts=starts, site_ids=np.asarray(["toy_a", "toy_b"]))
        json_write(output / "result.json", {"status": "smoke_ok", "metrics": metrics,
                                            "mean_training_loss": train_loss})
        expected = {"config.json", "environment.json", "checkpoint.pt", "predictions.npz", "result.json"}
        if not expected.issubset({path.name for path in output.iterdir()}):
            raise RuntimeError("smoke artifact check failed")
    print("smoke test passed (synthetic data, 2 updates, temporary output; no full training)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="synthetic-only quick validation")
    mode.add_argument("--run", action="store_true", help="run the requested external baseline experiments")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS,
                        help="repeatable; default is 13, 17, 29")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    seeds = tuple(args.seed) if args.seed else SEEDS
    destinations = [OUT_ROOT / dataset / f"seed_{seed}" for dataset in datasets for seed in seeds]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing run directories: " + ", ".join(map(str, existing)))
    if not CHARGED_ARCHIVE.is_file():
        raise FileNotFoundError(CHARGED_ARCHIVE)
    device = torch.device(args.device)
    source, mean, std, provenance = load_charged_source()
    for dataset in datasets:
        for seed in seeds:
            run_one(dataset, seed, source, mean, std, provenance, device)


if __name__ == "__main__":
    main()

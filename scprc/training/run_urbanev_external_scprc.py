#!/usr/bin/env python3
"""Frozen CHARGED-to-UrbanEV external generalization evaluation.

The source model is trained only on CHARGED April--July source windows using
 the pinned PhaseFormer and SCPRC-v2.2-PCGrad runners. UrbanEV September--
December values are used only as forecast history and, in a separate
unsupervised-scaling track, to fit a global target scaler. January is reported
as a development diagnostic; February is the final external test. No UrbanEV
forecast labels are used for training, release selection, or hyperparameter
selection.
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
import pandas as pd
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.train_scprc as v22  # noqa: E402
import scprc.training.source_calibration_support as cert  # noqa: E402
from scprc.data_adapters.charged.city_adapter import CITY_CODES  # noqa: E402
from scprc.models.scprc import (  # noqa: E402
    SCPRC,
)

CONTEXT = 168
HORIZON = 24
URBAN_TRAIN_END = 2928
URBAN_VAL_END = 3672
URBAN_TEST_END = 4344
SEED = 13
STEPS = 2048
TORCH_THREADS = 2
BATCH_SIZE = 16


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.DatetimeIndex):
        return [str(x) for x in value]
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_urbanev(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.shape != (URBAN_TEST_END, 276):
        raise ValueError(f"expected 4344x276 volume table, got {frame.shape}")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range("2022-09-01 00:00:00", periods=URBAN_TEST_END, freq="h")
    if not times.equals(expected):
        raise ValueError("UrbanEV timestamps are not the approved continuous hourly index")
    site_ids = tuple(str(x) for x in frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if len(site_ids) != 275 or len(set(site_ids)) != len(site_ids):
        raise ValueError("UrbanEV must have 275 unique zone columns")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("UrbanEV volume contains non-finite or negative values")
    return {
        "times": times,
        "site_ids": site_ids,
        "values": values,
        "rows": int(values.shape[0]),
        "sites": int(values.shape[1]),
        "source_path": str(path),
        "sha256": sha256(path),
    }


def make_source_support_cities(source_cities: dict[str, dict[str, Any]],
                               target_values: np.ndarray) -> dict[str, dict[str, Any]]:
    cities = dict(source_cities)
    cities["URBANEV"] = {"values": target_values}
    return cities


def target_scaler(values: np.ndarray) -> tuple[float, float]:
    prefix = values[:URBAN_TRAIN_END].astype(np.float64)
    mean = float(prefix.mean())
    std = float(prefix.std())
    return mean, max(std, 1e-6)


def starts_for(split: str) -> np.ndarray:
    if split == "january":
        return np.arange(URBAN_TRAIN_END, URBAN_VAL_END - HORIZON + 1, dtype=np.int64)
    if split == "february":
        return np.arange(URBAN_VAL_END, URBAN_TEST_END - HORIZON + 1, dtype=np.int64)
    raise ValueError(split)


def truth_for(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.stack([values[int(s):int(s) + HORIZON] for s in starts]).astype(np.float32)


def model_predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
                  mean: float, std: float, *, device: str) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            batch = starts[offset:offset + BATCH_SIZE]
            raw = np.stack([values[int(s) - CONTEXT:int(s)] for s in batch]).astype(np.float32)
            x = torch.from_numpy((raw - mean) / std).to(device)
            pred = model(x)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("non-finite external forecast")
            chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) * std + mean


def naive_predictions(values: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    last, daily, weekly = [], [], []
    for s in starts:
        s = int(s)
        history = values[s - CONTEXT:s]
        last.append(np.repeat(history[-1:, :], HORIZON, axis=0))
        daily.append(np.repeat(history[-24:, :].mean(axis=0, keepdims=True), HORIZON, axis=0))
        weekly.append(history[-168:-144, :])
    return {
        "last_value": np.asarray(last, dtype=np.float32),
        "daily_mean": np.asarray(daily, dtype=np.float32),
        "weekly_naive": np.asarray(weekly, dtype=np.float32),
    }


def score(pred: np.ndarray, truth: np.ndarray, train_std: float) -> dict[str, float]:
    error = pred.astype(np.float64) - truth.astype(np.float64)
    node_rmse = np.sqrt(np.mean(error * error, axis=(0, 1)))
    node_mae = np.mean(np.abs(error), axis=(0, 1))
    hourly_total_error = pred.astype(np.float64).sum(axis=2) - truth.astype(np.float64).sum(axis=2)
    window_rmse = np.sqrt(np.mean(error * error, axis=(1, 2)))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "macro_rmse": float(node_rmse.mean()),
        "macro_mae": float(node_mae.mean()),
        "total_rmse": float(np.sqrt(np.mean(hourly_total_error * hourly_total_error))),
        "total_mae": float(np.mean(np.abs(hourly_total_error))),
        "nrmse_train_prefix": float(np.sqrt(np.mean(error * error)) / max(train_std, 1e-6)),
        "nmae_train_prefix": float(np.mean(np.abs(error)) / max(train_std, 1e-6)),
        "window_rmse_p95": float(np.quantile(window_rmse, 0.95)),
        "windows": int(pred.shape[0]),
        "sites": int(pred.shape[2]),
    }


def build_candidate(n_sites: int, threshold: float, support_scale: float) -> torch.nn.Module:
    return SCPRC(
        channels=n_sites, seq_len=CONTEXT, pred_len=HORIZON, period_len=24,
        latent_dim=32, phase_layers=2, routers=1, heads=8,
        dropout=0.1, pos_embed=True, relation_mode="phase",
        relation_hidden=16, relation_rank=8,
        support_scale=support_scale,
        source_relation_threshold=threshold,
    )


def save_state(model: torch.nn.Module, path: Path) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


def load_models_from_checkpoints(source_cities: dict[str, dict[str, Any]], *, device: str,
                                 out: Path) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    source_ids = tuple(CITY_CODES)
    mean, std = protocol.source_scaler(source_cities, source_ids)
    threshold = cert._estimate_threshold(
        source_cities, source_ids, mean, std, seed=SEED,
        device=device, quantile=0.90,
    )
    profile_support_cities = make_source_support_cities(source_cities, URBAN_VALUES)
    support_scale, support_info = cert._source_support_scale(
        profile_support_cities, source_ids, "URBANEV",
    )
    backbone = protocol._build_model("phaseformer_exact", URBAN_SITES).float().to(device)
    candidate = build_candidate(URBAN_SITES, threshold, 1.0).float().to(device)
    backbone_path = out / "phaseformer_exact_source_all.pt"
    candidate_path = out / "scpcr_v22_pcgrad_candidate_diagnostic.pt"
    backbone.load_state_dict(torch.load(backbone_path, map_location=device)["state_dict"])
    candidate.load_state_dict(torch.load(candidate_path, map_location=device)["state_dict"])
    return backbone, candidate, {
        "source_ids": list(source_ids), "source_scaler_mean": mean,
        "source_scaler_std": std, "source_relation_threshold_q90": threshold,
        "source_support": support_info, "strict_release_support_scale": support_scale,
        "resumed_from_checkpoints": [str(backbone_path), str(candidate_path)],
    }


def train_models(source_cities: dict[str, dict[str, Any]], *, device: str,
                 out: Path) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    source_ids = tuple(CITY_CODES)
    mean, std = protocol.source_scaler(source_cities, source_ids)
    threshold = cert._estimate_threshold(
        source_cities, source_ids, mean, std, seed=SEED,
        device=device, quantile=0.90,
    )
    # The UrbanEV profile is measured before training and never uses Jan/Feb labels.
    profile_support_cities = make_source_support_cities(source_cities, URBAN_VALUES)
    support_scale, support_info = cert._source_support_scale(
        profile_support_cities, source_ids, "URBANEV",
    )
    started = time.perf_counter()
    backbone, backbone_info = protocol._train_one(
        "phaseformer_exact", source_cities, source_ids, URBAN_SITES,
        mean, std, seed=SEED, steps=STEPS, device=device,
        torch_threads=TORCH_THREADS,
    )
    save_state(backbone, out / "phaseformer_exact_source_all.pt")
    # Train the candidate as a diagnostic even if the fixed source-support
    # certificate will later select the exact backbone for release.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    candidate = build_candidate(URBAN_SITES, threshold, 1.0)
    candidate_info = v22._train_candidate_pcgrad(
        candidate, source_cities, source_ids, mean, std,
        seed=SEED, steps=STEPS, device=device,
        torch_threads=TORCH_THREADS,
    )
    save_state(candidate, out / "scpcr_v22_pcgrad_candidate_diagnostic.pt")
    info = {
        "source_ids": list(source_ids),
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "source_relation_threshold_q90": threshold,
        "source_support": support_info,
        "strict_release_support_scale": support_scale,
        "backbone": backbone_info,
        "candidate": candidate_info,
        "training_seconds_total": time.perf_counter() - started,
    }
    return backbone, candidate, info


def evaluate(models: dict[str, torch.nn.Module], urban: dict[str, Any], *, out: Path,
             source_scaler_values: tuple[float, float], target_scaler_values: tuple[float, float],
             device: str, alt_values: np.ndarray | None = None) -> dict[str, Any]:
    values = urban["values"]
    train_std = float(values[:URBAN_TRAIN_END].std())
    all_scores: dict[str, Any] = {}
    predictions_to_save: dict[str, np.ndarray] = {}
    for split in ("january", "february"):
        starts = starts_for(split)
        truth = truth_for(values, starts)
        split_preds: dict[str, np.ndarray] = naive_predictions(values, starts)
        for name, model in models.items():
            # Source scaling is the strict frozen-source inference track.
            split_preds[name + "_source_scale"] = model_predict(
                model, values, starts, *source_scaler_values, device=device,
            )
            # Target-prefix scaling is unsupervised target normalization; it is
            # reported separately and never used for release selection.
            split_preds[name + "_target_prefix_scale"] = model_predict(
                model, values, starts, *target_scaler_values, device=device,
            )
        scores = {name: score(pred, truth, train_std) for name, pred in split_preds.items()}
        all_scores[split] = scores
        for name, pred in split_preds.items():
            predictions_to_save[f"{split}__{name}"] = pred.astype(np.float32)
        if alt_values is not None:
            alt_truth = truth_for(alt_values, starts)
            alt_preds = naive_predictions(alt_values, starts)
            for name, model in models.items():
                alt_preds[name + "_source_scale"] = model_predict(
                    model, alt_values, starts, *source_scaler_values, device=device,
                )
            all_scores[split + "__volume_11kw_sensitivity"] = {
                name: score(pred, alt_truth, float(alt_values[:URBAN_TRAIN_END].std()))
                for name, pred in alt_preds.items()
            }
            for name, pred in alt_preds.items():
                predictions_to_save[f"{split}__volume_11kw__{name}"] = pred.astype(np.float32)
    np.savez_compressed(
        out / "predictions.npz",
        february_starts=starts_for("february"),
        february_timestamps=np.asarray([str(urban["times"][int(s)]) for s in starts_for("february")]),
        site_ids=np.asarray(urban["site_ids"], dtype=str),
        **predictions_to_save,
    )
    return all_scores


def main() -> int:
    global URBAN_VALUES, URBAN_SITES
    parser = argparse.ArgumentParser()
    parser.add_argument("--urban-volume", type=Path, default=ROOT / "research/urbanev_timemixer/sources/UrbanEV/data/volume.csv")
    parser.add_argument("--urban-volume-11kw", type=Path, default=ROOT / "research/urbanev_timemixer/sources/UrbanEV/data/volume-11kW.csv")
    parser.add_argument("--output", type=Path, default=HERE.parent / "urbanev_external_scpcr_v22_20261020")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="reuse source checkpoints already written in the output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urban = load_urbanev(args.urban_volume)
    URBAN_VALUES = urban["values"]
    URBAN_SITES = urban["sites"]
    if args.smoke_only:
        starts = starts_for("january")
        assert len(starts) == 721 and starts[-1] + HORIZON == URBAN_VAL_END
        print(json.dumps({"status": "passed", "urban_shape": list(URBAN_VALUES.shape), "jan_windows": len(starts)}))
        return 0
    # Load only the approved CHARGED prefixes. No UrbanEV labels are sent to this loader.
    source_cities, source_provenance = protocol.load_cities(protocol.ARCHIVE)
    mean_source, std_source = protocol.source_scaler(source_cities, tuple(CITY_CODES))
    mean_target, std_target = target_scaler(URBAN_VALUES)
    alt_values = None
    alt_hash = None
    if args.urban_volume_11kw.is_file():
        alt = load_urbanev(args.urban_volume_11kw)
        alt_values = alt["values"]
        alt_hash = alt["sha256"]
    (args.output / "run_status.json").write_text(json.dumps({
        "status": "training_started", "device": args.device,
        "urban_volume_sha256": urban["sha256"], "urban_volume_11kw_sha256": alt_hash,
        "source_provenance": source_provenance,
        "source_labels_used_for_training": "CHARGED Apr-Jul only",
        "urban_labels_used_for_training": False,
        "urban_january_or_february_labels_used_for_selection": False,
    }, indent=2) + "\n")
    checkpoint_ready = (args.output / "phaseformer_exact_source_all.pt").is_file() and (args.output / "scpcr_v22_pcgrad_candidate_diagnostic.pt").is_file()
    if args.reuse_checkpoints or checkpoint_ready:
        backbone, candidate, training = load_models_from_checkpoints(source_cities, device=args.device, out=args.output)
    else:
        backbone, candidate, training = train_models(source_cities, device=args.device, out=args.output)
    support_scale = float(training["strict_release_support_scale"])
    release = candidate if support_scale > 0.0 else backbone
    models = {
        "phaseformer_exact": backbone,
        "scpcr_v22_pcgrad_candidate_diagnostic": candidate,
        "scpcr_v22_pcgrad_strict_release": release,
    }
    scores = evaluate(
        models, urban, out=args.output,
        source_scaler_values=(mean_source, std_source),
        target_scaler_values=(mean_target, std_target),
        device=args.device, alt_values=alt_values,
    )
    result = {
        "status": "completed",
        "experiment": "Frozen CHARGED-to-UrbanEV external generalization",
        "model_protocol": "PhaseFormer exact and SCPRC-v2.2-PCGrad source-all-city retraining; no gate search, no expert pool",
        "seed": SEED, "steps": STEPS, "torch_threads": TORCH_THREADS,
        "urban_volume": {k: v for k, v in urban.items() if k not in ("times", "values")},
        "urban_split": {
            "prefix": "2022-09-01 00:00 through 2022-12-31 23:00",
            "development": "2023-01-01 00:00 through 2023-01-31 23:00",
            "final_test": "2023-02-01 00:00 through 2023-02-28 23:00",
            "context_hours": CONTEXT, "horizon_hours": HORIZON,
            "prefix_rows": URBAN_TRAIN_END, "development_rows": URBAN_VAL_END - URBAN_TRAIN_END,
            "test_rows": URBAN_TEST_END - URBAN_VAL_END,
        },
        "training": training,
        "strict_release_rule": "Use SCPRC candidate only when fixed source-support certificate passes; otherwise exact PhaseFormer fallback. No UrbanEV Jan/Feb labels are read by this rule.",
        "scalers": {
            "source_global": {"mean": mean_source, "std": std_source},
            "target_prefix_unsupervised": {"mean": mean_target, "std": std_target},
        },
        "scores": scores,
        "code_sha256": {
            "runner": sha256(HERE),
            "protocol": sha256(HERE.parent / "training_protocol.py"),
            "pcgrad": sha256(HERE.parent / "train_scprc.py"),
            "adapter": sha256(ROOT / "scprc/models/scprc.py"),
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n")
    (args.output / "run_status.json").write_text(json.dumps({"status": "completed", "result": str(args.output / "result.json")}, indent=2) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output), "strict_release": "scpcr_v22_pcgrad_strict_release", "scores": scores}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

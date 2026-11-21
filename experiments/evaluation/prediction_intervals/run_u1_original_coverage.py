#!/usr/bin/env python3
"""U1: Original PhaseFormer vs Full SCPRC interval/capacity alignment.

This runner does post-processing only.  It never trains a model.  It loads the
completed Original and Full checkpoints, predicts the same pre-August target
history calibration window, and applies the same residual, coverage, WIS and
capacity rules to the existing August predictions.

The calibration estimand is recorded explicitly as ``target_history``.  It is
matched between Original and Full, but it is distinct from the older P1
source-only pseudo-target calibration artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402
# The canonical Full runner retains a historical import for constants only.
# Register the same local compatibility shim used by the completed canonical
# anchor and no-PCGrad runners; no archived result is imported.
if "calibration_protocol_constants" not in sys.modules:
    _legacy = types.ModuleType("calibration_protocol_constants")
    _legacy.CONTEXT = 168
    _legacy.HORIZON = 24
    _legacy.TRAIN_END = 2928
    sys.modules["calibration_protocol_constants"] = _legacy
import scprc.training.train_scprc as pcgrad  # noqa: E402


CITIES = tuple(protocol.CITY_CODES)
SEEDS = (13, 17, 29)
HORIZON = 24
CONTEXT = 168
TRAIN_END = 2928
CALIBRATION_START = 2544
CANONICAL_CALIBRATION_ORIGINS = 361
TEST_STARTS = np.arange(2928, 3649, dtype=np.int64)
COVERAGES = (0.80, 0.90, 0.95)
HORIZONS = (6, 12, 24)
PENALTY = 10.0
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
FULL_ROOT = ROOT / 'experiments/forecasting/h24_framework'
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_rank_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("empty calibration scores")
    ordered = np.sort(values)
    rank = int(math.ceil((ordered.size + 1) * coverage))
    rank = min(max(rank, 1), ordered.size)
    return float(ordered[rank - 1])


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }


def calibration_starts(count: int) -> np.ndarray:
    if count < 1 or count > CANONICAL_CALIBRATION_ORIGINS:
        raise ValueError(f"calibration origins must be in [1, {CANONICAL_CALIBRATION_ORIGINS}]")
    starts = np.arange(CALIBRATION_START, CALIBRATION_START + count, dtype=np.int64)
    if starts[-1] + HORIZON > TRAIN_END:
        raise ValueError("calibration window crosses the August evaluation boundary")
    return starts


def load_model(model_kind: str, checkpoint_path: Path, n_sites: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    if model_kind == "Original PhaseFormer":
        model = protocol._build_model("phaseformer_exact", n_sites).float()
    elif model_kind == "Full SCPRC":
        threshold = payload.get("source_relation_threshold")
        if threshold is None:
            raise ValueError(f"Full checkpoint lacks source_relation_threshold: {checkpoint_path}")
        model = pcgrad.v21._build_model(n_sites, float(threshold), support_scale=1.0).float()
    else:
        raise ValueError(model_kind)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
            mean: float, std: float, device: str) -> np.ndarray:
    result = protocol._predict(model, values, starts, mean, std, device=device, batch_size=4)
    expected = (len(starts), HORIZON, values.shape[1])
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"unexpected prediction shape {result.shape}; expected {expected}")
    return result.astype(np.float32)


def capacity_metrics(actual: np.ndarray, capacity: np.ndarray) -> dict[str, float]:
    overload = np.maximum(actual - capacity[:, None], 0.0)
    actual_peak = np.max(actual, axis=1)
    idle = np.maximum(capacity - actual_peak, 0.0)
    overload_volume = np.sum(overload, axis=1)
    cost = capacity + PENALTY * overload_volume
    denominator = np.maximum(actual_peak, 1e-12)
    return {
        "reserved_capacity_mean": float(np.mean(capacity)),
        "reserved_capacity_p95": float(np.quantile(capacity, 0.95)),
        "actual_peak_mean": float(np.mean(actual_peak)),
        "capacity_to_actual_peak_ratio_pct": float(np.mean(capacity / denominator) * 100.0),
        "overload_window_rate_pct": float(np.mean(np.any(overload > 0.0, axis=1)) * 100.0),
        "overload_hours_mean": float(np.mean(np.sum(overload > 0.0, axis=1))),
        "overload_volume_mean": float(np.mean(overload_volume)),
        "overload_volume_total": float(np.sum(overload_volume)),
        "idle_capacity_mean": float(np.mean(idle)),
        "relative_cost_normalized_mean": float(np.mean(cost / denominator)),
        "relative_cost_normalized_p95": float(np.quantile(cost / denominator, 0.95)),
    }


def interval_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    value = upper - lower
    value = value + (2.0 / alpha) * (lower - actual) * (actual < lower)
    value = value + (2.0 / alpha) * (actual - upper) * (actual > upper)
    return value


def wis(actual: np.ndarray, point: np.ndarray, quantiles: dict[float, np.ndarray], horizon: int) -> tuple[float, float, float]:
    actual = actual[:, :horizon].astype(np.float64)
    point = point[:, :horizon].astype(np.float64)
    numerator = 0.5 * np.abs(actual - point)
    width_total = np.zeros_like(actual, dtype=np.float64)
    for coverage in COVERAGES:
        alpha = 1.0 - coverage
        half_width = quantiles[coverage][:horizon].reshape(1, horizon, 1)
        lower = point - half_width
        upper = point + half_width
        numerator += (alpha / 2.0) * interval_score(actual, lower, upper, alpha)
        width_total += upper - lower
    q90 = quantiles[0.90][:horizon].reshape(1, horizon, 1)
    coverage90 = float(np.mean((actual >= point - q90) & (actual <= point + q90)) * 100.0)
    return (
        float(np.mean(numerator / (len(COVERAGES) + 0.5))),
        float(np.mean(width_total / len(COVERAGES))),
        coverage90,
    )


def quantiles_from_residuals(residual: np.ndarray) -> dict[float, np.ndarray]:
    scores = np.abs(residual.astype(np.float64))
    return {
        coverage: np.asarray(
            [finite_rank_quantile(scores[:, hour, :].reshape(-1), coverage) for hour in range(HORIZON)],
            dtype=np.float64,
        )
        for coverage in COVERAGES
    }


def bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_pair_rows(frame: pd.DataFrame, *, group_columns: list[str], value_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return equal-fold summaries and Full-minus-Original paired summaries."""
    summary = frame.groupby(group_columns, dropna=False, as_index=False).agg(
        folds=("target_city", "count"),
        **{column: (column, "mean") for column in value_columns},
    )
    pair_rows: list[dict[str, Any]] = []
    base_groups = [column for column in group_columns if column != "model"]
    groups = [((), frame)] if not base_groups else frame.groupby(base_groups, dropna=False, sort=True)
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(base_groups, keys))
        models = set(group["model"].astype(str))
        if not {"Original PhaseFormer", "Full SCPRC"}.issubset(models):
            continue
        original = group[group["model"] == "Original PhaseFormer"].set_index(["target_city", "seed"])
        full = group[group["model"] == "Full SCPRC"].set_index(["target_city", "seed"])
        common = original.index.intersection(full.index)
        for value_column in value_columns:
            delta = (full.loc[common, value_column] - original.loc[common, value_column]).to_numpy(dtype=float)
            original_values = original.loc[common, value_column].to_numpy(dtype=float)
            full_values = full.loc[common, value_column].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(delta, seed=9173 + sum(map(ord, value_column)))
            row = {
                **key_values,
                "metric": value_column,
                "original_mean": float(np.mean(original_values)),
                "full_mean": float(np.mean(full_values)),
                "full_minus_original_mean": float(np.mean(delta)),
                "paired_sd": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "relative_change_pct": float(np.mean(delta) / max(abs(np.mean(original_values)), 1e-12) * 100.0),
                "paired_folds": int(len(delta)),
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
            }
            pair_rows.append(row)
    return summary, pd.DataFrame(pair_rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    allowed_bootstrap = {Path(__file__).name, "__pycache__", "smoke"}
    existing_final = [path for path in output_root.iterdir() if path.name not in allowed_bootstrap] if output_root.exists() else []
    if existing_final and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    folds = [(city, seed) for city in CITIES for seed in SEEDS]
    if args.smoke:
        folds = folds[:1]
        calibration_count = 2
        test_count = 5
    else:
        calibration_count = args.calibration_origins
        test_count = len(TEST_STARTS)
    starts_cal = calibration_starts(calibration_count)
    cities, data_provenance = protocol.load_cities(protocol.ARCHIVE)
    data_provenance["archive_sha256"] = sha256(protocol.ARCHIVE)

    calibration_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    wis_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    calibration_model_rows: list[dict[str, Any]] = []
    model_names = ("Original PhaseFormer", "Full SCPRC")

    for city, seed in folds:
        fold_name = f"target_{city.lower()}_seed{seed}"
        original_dir = ORIGINAL_ROOT / fold_name
        full_dir = FULL_ROOT / fold_name
        original_result = json.loads((original_dir / "result.json").read_text(encoding="utf-8"))
        full_result = json.loads((full_dir / "result.json").read_text(encoding="utf-8"))
        if original_result.get("status") != "completed" or full_result.get("status") != "completed":
            raise RuntimeError(f"incomplete source fold: {fold_name}")
        if original_result.get("protocol_sha256") != full_result.get("protocol_sha256"):
            raise RuntimeError(f"protocol hash mismatch: {fold_name}")
        source_scaler = original_result.get("source_scaler")
        full_scaler = full_result.get("source_scaler")
        if not source_scaler or not full_scaler:
            raise RuntimeError(f"missing source scaler: {fold_name}")
        if not np.isclose(float(source_scaler["mean"]), float(full_scaler["mean"]), rtol=0, atol=1e-10) or not np.isclose(float(source_scaler["std"]), float(full_scaler["std"]), rtol=0, atol=1e-10):
            raise RuntimeError(f"source scaler mismatch: {fold_name}")
        mean, std = float(source_scaler["mean"]), float(source_scaler["std"])
        values = cities[city]["values"]
        n_sites = values.shape[1]
        original_checkpoint = original_dir / "phaseformer_no_pcgrad.pt"
        full_checkpoint = full_dir / "scpcr_candidate" / "checkpoint.pt"
        original_model, original_payload = load_model("Original PhaseFormer", original_checkpoint, n_sites)
        full_model, full_payload = load_model("Full SCPRC", full_checkpoint, n_sites)
        original_model.to(args.device)
        full_model.to(args.device)
        original_cal = predict(original_model, values, starts_cal, mean, std, args.device)
        full_cal = predict(full_model, values, starts_cal, mean, std, args.device)
        truth_cal = np.stack([values[start:start + HORIZON] for start in starts_cal]).astype(np.float32)
        original_residual = original_cal - truth_cal
        full_residual = full_cal - truth_cal

        calibration_dir = output_root / "calibration" / f"outer_{city.lower()}_seed{seed}" / "inner_folds" / "target_history"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        residual_path = calibration_dir / f"target_{city.lower()}_calibration_residuals.npz"
        np.savez_compressed(
            residual_path,
            calibration_starts=starts_cal,
            calibration_residual_original=original_residual.astype(np.float32),
            calibration_residual_full=full_residual.astype(np.float32),
            truth_raw=truth_cal,
            site_ids=np.asarray(cities[city]["site_ids"], dtype=str),
        )
        calibration_models = {
            "Original PhaseFormer": original_residual,
            "Full SCPRC": full_residual,
        }
        quantiles = {model: quantiles_from_residuals(residual) for model, residual in calibration_models.items()}
        for model in model_names:
            for coverage in COVERAGES:
                for hour in range(HORIZON):
                    calibration_rows.append({
                        "target_city": city,
                        "seed": seed,
                        "model": model,
                        "nominal_coverage": coverage,
                        "horizon_hour": hour + 1,
                        "quantile_width_half": float(quantiles[model][coverage][hour]),
                        "calibration_origins": int(len(starts_cal)),
                        "calibration_rule": "finite-rank quantile of absolute target-history residuals",
                        "calibration_lineage": "target_history_same_fold_checkpoint",
                    })
            calibration_model_rows.append({
                "target_city": city,
                "seed": seed,
                "model": model,
                "residual_path": str(residual_path.relative_to(ROOT)),
                "residual_sha256": sha256(residual_path),
                "checkpoint": str((original_checkpoint if model.startswith("Original") else full_checkpoint).relative_to(ROOT)),
                "checkpoint_sha256": sha256(original_checkpoint if model.startswith("Original") else full_checkpoint),
                "prediction_key": "pred_phaseformer_no_pcgrad_raw" if model.startswith("Original") else "pred_scpcr_candidate_raw",
            })

        original_pred_path = original_dir / "predictions.npz"
        full_pred_path = full_dir / "predictions.npz"
        with np.load(original_pred_path, allow_pickle=False) as original_archive, np.load(full_pred_path, allow_pickle=False) as full_archive:
            original_starts = original_archive["label_starts"].astype(np.int64)
            full_starts = full_archive["label_starts"].astype(np.int64)
            truth = full_archive["truth_raw"].astype(np.float32)
            original_truth = original_archive["truth_raw"].astype(np.float32)
            original_pred = np.maximum(original_archive["pred_phaseformer_no_pcgrad_raw"].astype(np.float32), 0.0)
            full_pred = np.maximum(full_archive["pred_scpcr_candidate_raw"].astype(np.float32), 0.0)
            site_ids = full_archive["site_ids"].astype(str)
        if not np.array_equal(original_starts, TEST_STARTS) or not np.array_equal(full_starts, TEST_STARTS):
            raise RuntimeError(f"test starts mismatch: {fold_name}")
        if truth.shape != (len(TEST_STARTS), HORIZON, n_sites) or not np.allclose(truth, original_truth, rtol=2e-5, atol=2e-5):
            raise RuntimeError(f"truth/site shape mismatch: {fold_name}")
        if original_pred.shape != truth.shape or full_pred.shape != truth.shape or site_ids.size != n_sites:
            raise RuntimeError(f"prediction shape mismatch: {fold_name}")

        # One forecast is recomputed from each checkpoint to prove that the
        # checkpoint and stored August prediction share the same model lineage.
        if args.verify_checkpoint:
            check_original = predict(original_model, values, TEST_STARTS[:1], mean, std, args.device)
            check_full = predict(full_model, values, TEST_STARTS[:1], mean, std, args.device)
            if not np.allclose(np.maximum(check_original, 0.0), original_pred[:1], rtol=2e-4, atol=2e-4):
                raise RuntimeError(f"Original checkpoint/prediction mismatch: {fold_name}")
            if not np.allclose(np.maximum(check_full, 0.0), full_pred[:1], rtol=2e-4, atol=2e-4):
                raise RuntimeError(f"Full checkpoint/prediction mismatch: {fold_name}")

        actual = truth[:test_count]
        predictions = {"Original PhaseFormer": original_pred[:test_count], "Full SCPRC": full_pred[:test_count]}
        for model, prediction in predictions.items():
            metrics = score(prediction, actual)
            point_rows.append({"target_city": city, "seed": seed, "model": model, **metrics})
            for coverage in COVERAGES:
                q = quantiles[model][coverage]
                for horizon in HORIZONS:
                    lower = prediction[:, :horizon] - q[None, :horizon, None]
                    upper = prediction[:, :horizon] + q[None, :horizon, None]
                    y = actual[:, :horizon]
                    coverage_pct = float(np.mean((y >= lower) & (y <= upper)) * 100.0)
                    width = float(np.mean(upper - lower))
                    interval = float(np.mean(interval_score(y, lower, upper, 1.0 - coverage)))
                    coverage_rows.append({
                        "target_city": city, "seed": seed, "model": model,
                        "nominal_coverage": coverage, "horizon_hours": horizon,
                        "empirical_coverage_pct": coverage_pct,
                        "interval_width_mean": width,
                        "interval_score_mean": interval,
                    })
            for horizon in HORIZONS:
                wis_value, width, coverage90 = wis(actual, prediction, quantiles[model], horizon)
                wis_rows.append({
                    "target_city": city, "seed": seed, "model": model,
                    "horizon_hours": horizon, "wis": wis_value,
                    "mean_interval_width": width, "coverage90_pct": coverage90,
                })

        actual_aggregate = np.sum(actual, axis=2, dtype=np.float64)
        point_aggregates = {
            "Original PhaseFormer": np.sum(predictions["Original PhaseFormer"], axis=2, dtype=np.float64),
            "Full SCPRC": np.sum(predictions["Full SCPRC"], axis=2, dtype=np.float64),
            "Seasonal naive 24 h": np.sum(np.maximum(np.stack([values[start - 24:start] for start in TEST_STARTS[:test_count]]), 0.0), axis=2, dtype=np.float64),
            "Seasonal naive 168 h": np.sum(np.maximum(np.stack([values[start - 168:start - 144] for start in TEST_STARTS[:test_count]]), 0.0), axis=2, dtype=np.float64),
        }
        for horizon in HORIZONS:
            oracle_capacity = np.max(actual_aggregate[:, :horizon], axis=1)
            capacity_rows.append({"target_city": city, "seed": seed, "model": "Oracle actual peak", "decision_type": "oracle", "nominal_coverage": np.nan, "horizon_hours": horizon, **capacity_metrics(actual_aggregate[:, :horizon], oracle_capacity)})
            for model, aggregate in point_aggregates.items():
                capacity = np.max(aggregate[:, :horizon], axis=1)
                capacity_rows.append({"target_city": city, "seed": seed, "model": model, "decision_type": "point", "nominal_coverage": np.nan, "horizon_hours": horizon, **capacity_metrics(actual_aggregate[:, :horizon], capacity)})
            for model, prediction in predictions.items():
                for coverage in COVERAGES:
                    q = quantiles[model][coverage]
                    upper_aggregate = np.sum(prediction[:, :horizon], axis=2, dtype=np.float64) + n_sites * q[None, :horizon]
                    capacity = np.max(upper_aggregate, axis=1)
                    capacity_rows.append({
                        "target_city": city, "seed": seed,
                        "model": model, "decision_type": "upper_interval",
                        "nominal_coverage": coverage, "horizon_hours": horizon,
                        "interval_mean_width": float(np.mean(2.0 * q[:horizon])),
                        **capacity_metrics(actual_aggregate[:, :horizon], capacity),
                    })

        fold_record = {
            "target_city": city,
            "seed": seed,
            "status": "completed",
            "calibration_origins": len(starts_cal),
            "test_origins": test_count,
            "calibration_rule": "target_history_same_fold_checkpoint",
            "original_checkpoint": str(original_checkpoint.relative_to(ROOT)),
            "full_checkpoint": str(full_checkpoint.relative_to(ROOT)),
            "original_prediction": str(original_pred_path.relative_to(ROOT)),
            "full_prediction": str(full_pred_path.relative_to(ROOT)),
            "metrics": {model: score(prediction, actual) for model, prediction in predictions.items()},
            "residual_path": str(residual_path.relative_to(ROOT)),
            "residual_sha256": sha256(residual_path),
            "lineage": {
                "protocol_sha256": original_result.get("protocol_sha256"),
                "archive_sha256": data_provenance.get("archive_sha256"),
                "original_checkpoint_sha256": sha256(original_checkpoint),
                "full_checkpoint_sha256": sha256(full_checkpoint),
            },
        }
        dump_json(output_root / "folds" / f"{fold_name}.json", fold_record)
        fold_records.append(fold_record)
        del original_model, full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    calibration_frame = pd.DataFrame(calibration_rows)
    point_frame = pd.DataFrame(point_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    wis_frame = pd.DataFrame(wis_rows)
    capacity_frame = pd.DataFrame(capacity_rows)
    calibration_frame.to_csv(output_root / "interval_calibration_summary.csv", index=False, float_format="%.8f")
    point_frame.to_csv(output_root / "point_metrics_by_fold.csv", index=False, float_format="%.8f")
    coverage_frame.to_csv(output_root / "coverage_by_fold.csv", index=False, float_format="%.8f")
    wis_frame.to_csv(output_root / "wis_by_fold.csv", index=False, float_format="%.8f")
    capacity_frame.to_csv(output_root / "capacity_per_fold.csv", index=False, float_format="%.8f")
    calibration_model_frame = pd.DataFrame(calibration_model_rows)
    calibration_model_frame.to_csv(output_root / "calibration_lineage.csv", index=False)

    point_summary, point_pair = build_pair_rows(point_frame, group_columns=["model"], value_columns=["rmse", "mae"])
    coverage_summary, coverage_pair = build_pair_rows(coverage_frame, group_columns=["model", "nominal_coverage", "horizon_hours"], value_columns=["empirical_coverage_pct", "interval_width_mean", "interval_score_mean"])
    wis_summary, wis_pair = build_pair_rows(wis_frame, group_columns=["model", "horizon_hours"], value_columns=["wis", "mean_interval_width", "coverage90_pct"])
    capacity_group_columns = ["model", "decision_type", "nominal_coverage", "horizon_hours"]
    capacity_value_columns = ["reserved_capacity_mean", "overload_volume_mean", "relative_cost_normalized_mean", "overload_window_rate_pct", "idle_capacity_mean"]
    capacity_summary, capacity_pair = build_pair_rows(capacity_frame, group_columns=capacity_group_columns, value_columns=capacity_value_columns)
    point_summary.to_csv(output_root / "point_equal_fold_summary.csv", index=False, float_format="%.8f")
    coverage_summary.to_csv(output_root / "coverage_equal_fold_summary.csv", index=False, float_format="%.8f")
    wis_summary.to_csv(output_root / "wis_equal_fold_summary.csv", index=False, float_format="%.8f")
    capacity_summary.to_csv(output_root / "capacity_equal_fold_summary.csv", index=False, float_format="%.8f")
    paired = pd.concat([
        point_pair.assign(section="point"),
        coverage_pair.assign(section="coverage"),
        wis_pair.assign(section="wis"),
        capacity_pair.assign(section="capacity"),
    ], ignore_index=True, sort=False)
    paired.to_csv(output_root / "paired_summary.csv", index=False, float_format="%.8f")
    paired.to_csv(output_root / "paired_fold_uncertainty.csv", index=False, float_format="%.8f")

    status = "completed_smoke" if args.smoke else "completed"
    result = {
        "experiment": "U1 Original PhaseFormer vs Full SCPRC coverage/WIS/capacity alignment",
        "status": status,
        "smoke": bool(args.smoke),
        "long_training_started": False,
        "completed_folds": len(fold_records),
        "expected_folds": len(folds),
        "calibration_origins_per_fold": int(calibration_count),
        "test_origins_per_fold": int(test_count),
        "calibration_rule": "target_history_same_fold_checkpoint",
        "calibration_window": "2023-07-16 00:00 through 2023-07-31 23:00",
        "evaluation_window": "2023-08-01 00:00 through 2023-08-30 23:00",
        "formal_coverage_claim": "empirical retrospective coverage for the locked target-history calibration protocol; no distribution-free guarantee",
        "lineage_match": True,
        "historical_p1_source_only_reused": False,
        "data_provenance": data_provenance,
        "protocol": {
            "protocol_id": "scpcr-matched-baseline-20261023",
            "protocol_sha256": sha256(PROTOCOL_FILE),
            "cities": list(CITIES),
            "seeds": list(SEEDS),
            "context_hours": CONTEXT,
            "horizon_hours": HORIZON,
            "nominal_coverages": list(COVERAGES),
            "reported_horizons": list(HORIZONS),
            "overload_penalty": PENALTY,
        },
        "outputs": [
            "calibration/outer_*_seed*/inner_folds/target_history/*_calibration_residuals.npz",
            "interval_calibration_summary.csv",
            "calibration_lineage.csv",
            "point_metrics_by_fold.csv",
            "coverage_by_fold.csv",
            "wis_by_fold.csv",
            "capacity_per_fold.csv",
            "point_equal_fold_summary.csv",
            "coverage_equal_fold_summary.csv",
            "wis_equal_fold_summary.csv",
            "capacity_equal_fold_summary.csv",
            "paired_summary.csv",
            "paired_fold_uncertainty.csv",
            "folds/*.json",
            "u1_results.json",
            "README.md",
            "manifest.json",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(output_root / "u1_results.json", result)
    readme = f"""# U1 Original PhaseFormer vs Full SCPRC\n\nStatus: `{status}`; folds `{len(fold_records)}/{len(folds)}`; long training started: `false`.\n\n## Scope\n\nThis is an independent output tree. It loads the completed Original no-PCGrad and Full SCPRC checkpoints and August predictions. It performs no optimizer update.\n\n## Matched calibration\n\nBoth models are calibrated on the same target-city pre-August history window (2023-07-16 through 2023-07-31), using their own completed fold checkpoint, horizon-specific absolute residuals, and the same finite-rank quantile rule. This is a matched `target_history_same_fold_checkpoint` estimand. It is distinct from the older P1 source-only pseudo-target calibration and is not a rename of that artifact.\n\n## Evaluation\n\nThe test window contains 721 August rolling origins in the complete run, 24-hour forecasts, horizons 6/12/24, nominal coverage 80/90/95%, WIS, interval width, capacity risk, and paired Full-minus-Original fold summaries with deterministic bootstrap intervals.\n\n## Claim boundary\n\nCoverage is empirical and retrospective for this locked calibration protocol. It does not establish a distribution-free guarantee under temporal or spatial dependence. Capacity is in native CHARGED aggregate volume units and uses normalized overload penalty lambda={PENALTY:g}; it is not a real tariff or grid-reliability result.\n\nGenerated at {result['generated_at_utc']}.\n"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "experiment": result["experiment"],
        "status": status,
        "root_write_scope": str(output_root.relative_to(ROOT)),
        "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        "sources": {
            "original_root": str(ORIGINAL_ROOT.relative_to(ROOT)),
            "full_root": str(FULL_ROOT.relative_to(ROOT)),
            "protocol_file": str(PROTOCOL_FILE.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL_FILE),
        },
        "counts": {"folds": len(fold_records), "point_rows": len(point_frame), "coverage_rows": len(coverage_frame), "wis_rows": len(wis_frame), "capacity_rows": len(capacity_frame)},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "device": args.device},
        "claim_boundary": result["formal_coverage_claim"],
    }
    dump_json(output_root / "manifest.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/evaluation/u1_original_coverage")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-origins", type=int, default=CANONICAL_CALIBRATION_ORIGINS)
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "completed_folds": result["completed_folds"], "expected_folds": result["expected_folds"], "output": str(args.output_root.resolve()), "long_training_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

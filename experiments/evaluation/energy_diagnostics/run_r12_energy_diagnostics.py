#!/usr/bin/env python3
"""R12: energy-operation diagnostics from canonical Original/Full forecasts.

This is a read-only post-processing experiment.  It does not train, tune, or
copy models/data.  The only files written by this program are under the
directory containing this script.

The independent reporting unit is one city x seed fold.  Rolling origins and
stations are repeated observations nested inside that fold.  The script
therefore emits both origin-level diagnostics and equal-fold summaries; it
does not treat all hourly cells as independent experimental replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "scprc/data_adapters/charged/Hourly.zip"
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
FULL_ROOT = ROOT / 'experiments/forecasting/h24_framework'

CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
STARTS = np.arange(2928, 3649, dtype=np.int64)
HORIZON = 24
CONTEXT = 168
TRAIN_END = 2928
N_ORIGINS = 721
MARGINS = (0.00, 0.05, 0.10, 0.20)
PENALTIES = (1.0, 5.0, 10.0, 20.0)
PRIMARY_MARGIN = 0.10
PRIMARY_PENALTY = 10.0
RAMP_QUANTILE = 0.90

MODEL_ORDER = (
    "Original PhaseFormer",
    "Full SCPRC",
    "Seasonal naive 24 h",
    "Seasonal naive 168 h",
)
PREDICTION_KEYS = {
    "Original PhaseFormer": "pred_phaseformer_no_pcgrad_raw",
    "Full SCPRC": "pred_scpcr_candidate_raw",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_city(city: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read the same prefix-only hourly grid used by the prior replay."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"{city}/volume.csv") as stream:
            frame = pd.read_csv(stream, nrows=3672)
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="raise"))
    expected = pd.date_range("2023-04-01 00:00:00", periods=3672, freq="h")
    if not times.equals(expected):
        raise ValueError(f"{city}: unexpected hourly grid")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.shape[0] != 3672 or not np.isfinite(values).all():
        raise ValueError(f"{city}: invalid volume values")
    return values, tuple(str(value) for value in frame.columns[1:])


def load_prediction(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        starts = archive["label_starts"].astype(np.int64)
        truth = archive["truth_raw"].astype(np.float64)
        site_ids = tuple(str(value) for value in archive["site_ids"])
        prediction = archive[key].astype(np.float64)
    if not np.array_equal(starts, STARTS):
        raise ValueError(f"{path}: label_starts do not match canonical [2928, 3649)")
    expected = (N_ORIGINS, HORIZON, len(site_ids))
    if truth.shape != expected or prediction.shape != expected:
        raise ValueError(f"{path}: shape mismatch, truth={truth.shape}, prediction={prediction.shape}, expected={expected}")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError(f"{path}: non-finite truth/prediction")
    return truth, np.maximum(prediction, 0.0), starts, site_ids


def load_fold(city: str, seed: int, city_values: np.ndarray, site_ids: tuple[str, ...]) -> dict[str, Any]:
    fold = f"target_{city.lower()}_seed{seed}"
    original_path = ORIGINAL_ROOT / fold / "predictions.npz"
    full_path = FULL_ROOT / fold / "predictions.npz"
    truth_original, original, starts_original, sites_original = load_prediction(
        original_path, PREDICTION_KEYS["Original PhaseFormer"]
    )
    truth_full, full, starts_full, sites_full = load_prediction(full_path, PREDICTION_KEYS["Full SCPRC"])
    if not np.array_equal(starts_original, starts_full) or not np.allclose(truth_original, truth_full, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{fold}: Original/Full labels differ")
    if sites_original != sites_full or sites_original != site_ids:
        raise ValueError(f"{fold}: site identifiers differ from canonical archive")
    archive_truth = np.stack([city_values[start : start + HORIZON] for start in STARTS], axis=0).astype(np.float64)
    if not np.allclose(archive_truth, truth_full, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{fold}: prediction truth does not match CHARGED archive")
    seasonal24 = np.stack([city_values[start - 24 : start, :] for start in STARTS], axis=0).astype(np.float64)
    seasonal168 = np.stack([city_values[start - 168 : start - 144, :] for start in STARTS], axis=0).astype(np.float64)
    return {
        "city": city,
        "seed": seed,
        "fold": fold,
        "truth": truth_full,
        "models": {
            "Original PhaseFormer": original,
            "Full SCPRC": full,
            "Seasonal naive 24 h": np.maximum(seasonal24, 0.0),
            "Seasonal naive 168 h": np.maximum(seasonal168, 0.0),
        },
        "original_prediction_path": original_path,
        "full_prediction_path": full_path,
        "original_prediction_sha256": sha256(original_path),
        "full_prediction_sha256": sha256(full_path),
        "site_ids": site_ids,
        "city_values": city_values,
    }


def threshold_from_training(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Return city-total and station-wise 90th percentile absolute ramps.

    The threshold uses only the pre-August observed prefix (through index
    TRAIN_END), never August labels.  It is fixed within each city and is
    reported as a stress threshold, not a learned operating limit.
    """
    prefix = values[:TRAIN_END].astype(np.float64)
    station_threshold = np.quantile(np.abs(np.diff(prefix, axis=0)), RAMP_QUANTILE, axis=0)
    city_threshold = float(np.quantile(np.abs(np.diff(prefix.sum(axis=1))), RAMP_QUANTILE))
    return city_threshold, station_threshold


def _safe_relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator = np.asarray(numerator, dtype=np.float64)
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-12
    result[valid] = numerator[valid] / np.abs(denominator[valid]) * 100.0
    return result


def operational_metrics(actual: np.ndarray, prediction: np.ndarray, threshold: float | np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute peak, timing, ramp, and 24-hour volume metrics.

    Inputs are [origins, hours, series].  A city-total call has one series;
    the station-level call has one series per charging station.
    """
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.ndim == 2:
        actual = actual[:, :, None]
    if prediction.ndim == 2:
        prediction = prediction[:, :, None]
    if actual.shape != prediction.shape:
        raise ValueError(f"operational metric shape mismatch: {actual.shape} vs {prediction.shape}")
    origins, hours, series = actual.shape

    actual_peak = np.max(actual, axis=1)
    prediction_peak = np.max(prediction, axis=1)
    peak_signed = prediction_peak - actual_peak
    peak_abs = np.abs(peak_signed)
    peak_relative = _safe_relative(peak_abs, actual_peak)
    actual_peak_hour = np.argmax(actual, axis=1)
    prediction_peak_hour = np.argmax(prediction, axis=1)
    timing_signed = prediction_peak_hour - actual_peak_hour
    timing_abs = np.abs(timing_signed)

    actual_ramp = np.diff(actual, axis=1)
    prediction_ramp = np.diff(prediction, axis=1)
    ramp_error = prediction_ramp - actual_ramp
    ramp_abs = np.abs(ramp_error)
    if np.isscalar(threshold):
        threshold_array = np.full((series,), float(threshold), dtype=np.float64)
    else:
        threshold_array = np.asarray(threshold, dtype=np.float64).reshape(series)
    actual_exceed = np.abs(actual_ramp) > threshold_array.reshape(1, 1, series)
    prediction_exceed = np.abs(prediction_ramp) > threshold_array.reshape(1, 1, series)
    actual_exceed_count = actual_exceed.sum(axis=(1, 2))
    prediction_exceed_count = prediction_exceed.sum(axis=(1, 2))
    joint_exceed_count = (actual_exceed & prediction_exceed).sum(axis=(1, 2))
    actual_exceed_denominator = np.maximum(actual_exceed_count, 1)

    actual_total = np.sum(actual, axis=1)
    prediction_total = np.sum(prediction, axis=1)
    total_signed = prediction_total - actual_total
    total_abs = np.abs(total_signed)
    total_relative = _safe_relative(total_abs, actual_total)

    per_origin = {
        "actual_peak": actual_peak,
        "prediction_peak": prediction_peak,
        "peak_signed_error": peak_signed,
        "peak_abs_error": peak_abs,
        "peak_relative_abs_error_pct": peak_relative,
        "actual_peak_hour": actual_peak_hour,
        "prediction_peak_hour": prediction_peak_hour,
        "peak_timing_signed_hours": timing_signed,
        "peak_timing_abs_hours": timing_abs,
        "ramp_mae": np.mean(ramp_abs, axis=(1, 2)),
        "actual_ramp_exceedance_rate_pct": np.mean(actual_exceed, axis=(1, 2)) * 100.0,
        "prediction_ramp_exceedance_rate_pct": np.mean(prediction_exceed, axis=(1, 2)) * 100.0,
        "ramp_exceedance_rate_gap_pct": (
            np.mean(prediction_exceed, axis=(1, 2)) - np.mean(actual_exceed, axis=(1, 2))
        ) * 100.0,
        "ramp_exceedance_hit_rate_pct": joint_exceed_count / actual_exceed_denominator * 100.0,
        "actual_ramp_exceedance_count": actual_exceed_count,
        "prediction_ramp_exceedance_count": prediction_exceed_count,
        "total_volume_actual": actual_total,
        "total_volume_prediction": prediction_total,
        "total_volume_signed_error": total_signed,
        "total_volume_abs_error": total_abs,
        "total_volume_relative_abs_error_pct": total_relative,
    }
    summary = {
        "origin_count": float(origins),
        "series_count": float(series),
        "peak_level_mae": float(np.mean(peak_abs)),
        "peak_level_bias": float(np.mean(peak_signed)),
        "peak_level_mape_pct_positive_actual": float(np.nanmean(peak_relative)),
        "peak_level_mape_valid_rate_pct": float(np.mean(np.isfinite(peak_relative)) * 100.0),
        "peak_level_wape_pct": float(np.sum(peak_abs) / max(np.sum(actual_peak), 1e-12) * 100.0),
        "peak_timing_mae_hours": float(np.mean(timing_abs)),
        "peak_timing_bias_hours": float(np.mean(timing_signed)),
        "peak_timing_exact_rate_pct": float(np.mean(timing_abs == 0) * 100.0),
        "ramp_mae": float(np.mean(ramp_abs)),
        "actual_ramp_exceedance_rate_pct": float(np.mean(actual_exceed) * 100.0),
        "prediction_ramp_exceedance_rate_pct": float(np.mean(prediction_exceed) * 100.0),
        "ramp_exceedance_rate_gap_pct": float((np.mean(prediction_exceed) - np.mean(actual_exceed)) * 100.0),
        "ramp_exceedance_hit_rate_pct": float(np.sum(joint_exceed_count) / max(np.sum(actual_exceed_count), 1) * 100.0),
        "total_volume_mae": float(np.mean(total_abs)),
        "total_volume_bias": float(np.mean(total_signed)),
        "total_volume_mape_pct_positive_actual": float(np.nanmean(total_relative)),
        "total_volume_mape_valid_rate_pct": float(np.mean(np.isfinite(total_relative)) * 100.0),
        "total_volume_wape_pct": float(np.sum(total_abs) / max(np.sum(actual_total), 1e-12) * 100.0),
        "ramp_threshold_q90_raw_units_mean": float(np.mean(threshold_array)),
        "ramp_threshold_q90_raw_units_p95": float(np.quantile(threshold_array, 0.95)),
    }
    return summary, per_origin


def capacity_metrics(actual_aggregate: np.ndarray, prediction_aggregate: np.ndarray, margin: float, penalty: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Apply the frozen normalized capacity replay rule to one fold."""
    capacity = np.max(prediction_aggregate, axis=1) * (1.0 + margin)
    actual_peak = np.max(actual_aggregate, axis=1)
    overload = np.maximum(actual_aggregate - capacity[:, None], 0.0)
    overload_mask = overload > 0.0
    idle = np.maximum(capacity - actual_peak, 0.0)
    overload_volume = np.sum(overload, axis=1)
    cost_raw = capacity + penalty * overload_volume
    denominator = np.maximum(actual_peak, 1e-12)
    cost_normalized = cost_raw / denominator
    origin = {
        "reserved_capacity": capacity,
        "actual_peak": actual_peak,
        "capacity_to_actual_peak_ratio_pct": capacity / denominator * 100.0,
        "overload_window": np.any(overload_mask, axis=1).astype(np.int8),
        "overload_hours": np.sum(overload_mask, axis=1),
        "overload_volume": overload_volume,
        "idle_capacity": idle,
        "overreserve_ratio_pct": idle / denominator * 100.0,
        "actual_energy": np.sum(actual_aggregate, axis=1),
        "relative_cost_raw": cost_raw,
        "relative_cost_normalized": cost_normalized,
    }
    summary = {
        "origins": float(len(capacity)),
        "reserved_capacity_mean": float(np.mean(capacity)),
        "reserved_capacity_p95": float(np.quantile(capacity, 0.95)),
        "actual_peak_mean": float(np.mean(actual_peak)),
        "actual_peak_p95": float(np.quantile(actual_peak, 0.95)),
        "capacity_to_actual_peak_ratio_pct": float(np.mean(capacity / denominator) * 100.0),
        "overload_window_rate_pct": float(np.mean(np.any(overload_mask, axis=1)) * 100.0),
        "overload_hours_mean": float(np.mean(np.sum(overload_mask, axis=1))),
        "overload_hours_total": float(np.sum(overload_mask)),
        "overload_volume_mean": float(np.mean(overload_volume)),
        "overload_volume_total": float(np.sum(overload_volume)),
        "idle_capacity_mean": float(np.mean(idle)),
        "idle_capacity_p95": float(np.quantile(idle, 0.95)),
        "overreserve_ratio_pct": float(np.mean(idle / denominator) * 100.0),
        "actual_energy_mean": float(np.mean(np.sum(actual_aggregate, axis=1))),
        "relative_cost_raw_mean": float(np.mean(cost_raw)),
        "relative_cost_normalized_mean": float(np.mean(cost_normalized)),
        "relative_cost_normalized_p95": float(np.quantile(cost_normalized, 0.95)),
    }
    return summary, origin


def oracle_capacity(actual_aggregate: np.ndarray, margin: float, penalty: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Unavailable diagnostic lower bound: reserve on realized peak."""
    capacity = np.max(actual_aggregate, axis=1)
    actual_peak = capacity.copy()
    zeros = np.zeros_like(capacity)
    origin = {
        "reserved_capacity": capacity,
        "actual_peak": actual_peak,
        "capacity_to_actual_peak_ratio_pct": np.full_like(capacity, 100.0),
        "overload_window": np.zeros_like(capacity, dtype=np.int8),
        "overload_hours": zeros,
        "overload_volume": zeros,
        "idle_capacity": zeros,
        "overreserve_ratio_pct": zeros,
        "actual_energy": np.sum(actual_aggregate, axis=1),
        "relative_cost_raw": capacity,
        "relative_cost_normalized": np.ones_like(capacity),
    }
    summary = {
        "origins": float(len(capacity)),
        "reserved_capacity_mean": float(np.mean(capacity)),
        "reserved_capacity_p95": float(np.quantile(capacity, 0.95)),
        "actual_peak_mean": float(np.mean(actual_peak)),
        "actual_peak_p95": float(np.quantile(actual_peak, 0.95)),
        "capacity_to_actual_peak_ratio_pct": 100.0,
        "overload_window_rate_pct": 0.0,
        "overload_hours_mean": 0.0,
        "overload_hours_total": 0.0,
        "overload_volume_mean": 0.0,
        "overload_volume_total": 0.0,
        "idle_capacity_mean": 0.0,
        "idle_capacity_p95": 0.0,
        "overreserve_ratio_pct": 0.0,
        "actual_energy_mean": float(np.mean(np.sum(actual_aggregate, axis=1))),
        "relative_cost_raw_mean": float(np.mean(capacity)),
        "relative_cost_normalized_mean": 1.0,
        "relative_cost_normalized_p95": 1.0,
    }
    return summary, origin


def summary_row(city: str, seed: int, fold: str, model: str, layer: str, metrics: dict[str, float], threshold: float | np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"target_city": city, "seed": seed, "fold": fold, "model": model, "layer": layer}
    row.update(metrics)
    return row


def equal_fold_mean(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Average only measured numeric outcomes and retain fold count.

    Seed and origin counts are design metadata, not outcomes; averaging them
    into a manuscript table would make the summary harder to interpret.
    """
    value_columns = [
        column
        for column in frame.columns
        if column not in set(group_columns) | {"target_city", "seed", "fold", "origin_count", "series_count"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = frame.groupby(group_columns, as_index=False)[value_columns].mean()
    counts = frame.groupby(group_columns, as_index=False)["fold"].nunique().rename(columns={"fold": "fold_count"})
    return summary.merge(counts, on=group_columns, how="left")


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    city_cache = {city: read_city(city) for city in CITIES}
    folds: list[dict[str, Any]] = []
    operational_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    capacity_origin_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for city in CITIES:
        city_values, site_ids = city_cache[city]
        city_threshold, station_threshold = threshold_from_training(city_values)
        threshold_rows.append({
            "target_city": city,
            "ramp_threshold_definition": f"q{int(RAMP_QUANTILE * 100)} of absolute pre-August hourly ramp",
            "city_total_threshold_raw_units": city_threshold,
            "station_threshold_mean_raw_units": float(np.mean(station_threshold)),
            "station_threshold_p95_raw_units": float(np.quantile(station_threshold, 0.95)),
            "station_count": len(site_ids),
            "training_prefix_end_exclusive_index": TRAIN_END,
        })
        for seed in SEEDS:
            fold_data = load_fold(city, seed, city_values, site_ids)
            folds.append(fold_data)
            actual = fold_data["truth"]
            for model in MODEL_ORDER:
                prediction = fold_data["models"][model]
                actual_city = np.sum(actual, axis=2)
                prediction_city = np.sum(prediction, axis=2)
                city_metrics, city_origin = operational_metrics(actual_city, prediction_city, city_threshold)
                operational_rows.append(summary_row(city, seed, fold_data["fold"], model, "city_total", city_metrics, city_threshold))
                for origin_idx in range(N_ORIGINS):
                    origin_rows.append({
                        "target_city": city,
                        "seed": seed,
                        "fold": fold_data["fold"],
                        "model": model,
                        "layer": "city_total",
                        "origin_index": origin_idx,
                        "label_start": int(STARTS[origin_idx]),
                        "peak_level_abs_error": float(city_origin["peak_abs_error"][origin_idx, 0]),
                        "peak_level_signed_error": float(city_origin["peak_signed_error"][origin_idx, 0]),
                        "peak_timing_abs_hours": float(city_origin["peak_timing_abs_hours"][origin_idx, 0]),
                        "peak_timing_signed_hours": float(city_origin["peak_timing_signed_hours"][origin_idx, 0]),
                        "ramp_mae": float(city_origin["ramp_mae"][origin_idx]),
                        "actual_ramp_exceedance_rate_pct": float(city_origin["actual_ramp_exceedance_rate_pct"][origin_idx]),
                        "prediction_ramp_exceedance_rate_pct": float(city_origin["prediction_ramp_exceedance_rate_pct"][origin_idx]),
                        "ramp_exceedance_rate_gap_pct": float(city_origin["ramp_exceedance_rate_gap_pct"][origin_idx]),
                        "ramp_exceedance_hit_rate_pct": float(city_origin["ramp_exceedance_hit_rate_pct"][origin_idx]),
                        "total_volume_abs_error": float(city_origin["total_volume_abs_error"][origin_idx, 0]),
                        "total_volume_signed_error": float(city_origin["total_volume_signed_error"][origin_idx, 0]),
                        "total_volume_relative_abs_error_pct": float(city_origin["total_volume_relative_abs_error_pct"][origin_idx, 0]),
                        "actual_peak": float(city_origin["actual_peak"][origin_idx, 0]),
                        "prediction_peak": float(city_origin["prediction_peak"][origin_idx, 0]),
                        "actual_total_volume": float(city_origin["total_volume_actual"][origin_idx, 0]),
                        "prediction_total_volume": float(city_origin["total_volume_prediction"][origin_idx, 0]),
                    })
                station_metrics, _ = operational_metrics(actual, prediction, station_threshold)
                operational_rows.append(summary_row(city, seed, fold_data["fold"], model, "station_level", station_metrics, station_threshold))

            actual_city = np.sum(actual, axis=2)
            for margin in MARGINS:
                for penalty in PENALTIES:
                    for model in MODEL_ORDER:
                        prediction_city = np.sum(fold_data["models"][model], axis=2)
                        metrics, origin = capacity_metrics(actual_city, prediction_city, margin, penalty)
                        decision_type = "point"
                        capacity_rows.append({
                            "target_city": city,
                            "seed": seed,
                            "fold": fold_data["fold"],
                            "model": model,
                            "decision_type": decision_type,
                            "safety_margin_pct": margin * 100.0,
                            "overload_penalty": penalty,
                            **metrics,
                        })
                        for origin_idx in range(N_ORIGINS):
                            capacity_origin_rows.append({
                                "target_city": city,
                                "seed": seed,
                                "fold": fold_data["fold"],
                                "model": model,
                                "decision_type": decision_type,
                                "safety_margin_pct": margin * 100.0,
                                "overload_penalty": penalty,
                                "origin_index": origin_idx,
                                "label_start": int(STARTS[origin_idx]),
                                **{key: float(value[origin_idx]) for key, value in origin.items()},
                            })
                    metrics, origin = oracle_capacity(actual_city, margin, penalty)
                    capacity_rows.append({
                        "target_city": city,
                        "seed": seed,
                        "fold": fold_data["fold"],
                        "model": "Oracle actual peak",
                        "decision_type": "oracle_unavailable",
                        "safety_margin_pct": margin * 100.0,
                        "overload_penalty": penalty,
                        **metrics,
                    })
                    for origin_idx in range(N_ORIGINS):
                        capacity_origin_rows.append({
                            "target_city": city,
                            "seed": seed,
                            "fold": fold_data["fold"],
                            "model": "Oracle actual peak",
                            "decision_type": "oracle_unavailable",
                            "safety_margin_pct": margin * 100.0,
                            "overload_penalty": penalty,
                            "origin_index": origin_idx,
                            "label_start": int(STARTS[origin_idx]),
                            **{key: float(value[origin_idx]) for key, value in origin.items()},
                        })

    operational = pd.DataFrame(operational_rows)
    origins = pd.DataFrame(origin_rows)
    capacities = pd.DataFrame(capacity_rows)
    capacity_origins = pd.DataFrame(capacity_origin_rows)
    thresholds = pd.DataFrame(threshold_rows)

    operational.to_csv(OUT / "operational_metrics_per_fold.csv", index=False, float_format="%.8f")
    origins.to_csv(OUT / "operational_metrics_per_origin_city_total.csv", index=False, float_format="%.8f")
    thresholds.to_csv(OUT / "ramp_thresholds_by_city.csv", index=False, float_format="%.8f")
    capacities.to_csv(OUT / "capacity_reserve_risk_per_fold.csv", index=False, float_format="%.8f")
    capacity_origins.to_csv(OUT / "capacity_reserve_risk_per_origin.csv", index=False, float_format="%.8f")

    primary = capacities[
        (capacities["safety_margin_pct"] == PRIMARY_MARGIN * 100.0)
        & (capacities["overload_penalty"] == PRIMARY_PENALTY)
    ].copy()
    primary_summary = equal_fold_mean(primary, ["model", "decision_type"])
    primary_summary.to_csv(OUT / "capacity_reserve_risk_primary_equal_fold.csv", index=False, float_format="%.8f")

    equal_fold_operational = equal_fold_mean(operational, ["model", "layer"])
    equal_fold_operational.to_csv(OUT / "operational_metrics_equal_fold.csv", index=False, float_format="%.8f")
    equal_fold_capacity = equal_fold_mean(capacities, ["model", "decision_type", "safety_margin_pct", "overload_penalty"])
    equal_fold_capacity.to_csv(OUT / "capacity_reserve_risk_equal_fold.csv", index=False, float_format="%.8f")

    prediction_files = {}
    for fold in folds:
        prediction_files[fold["fold"]] = {
            "original": str(fold["original_prediction_path"].relative_to(ROOT)),
            "original_sha256": fold["original_prediction_sha256"],
            "full": str(fold["full_prediction_path"].relative_to(ROOT)),
            "full_sha256": fold["full_prediction_sha256"],
        }
    manifest = {
        "status": "completed",
        "experiment": "R12 canonical energy-operation diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_postprocessing": True,
        "no_retraining": True,
        "cities": list(CITIES),
        "seeds": list(SEEDS),
        "fold_count": len(folds),
        "origins_per_fold": N_ORIGINS,
        "horizon_hours": HORIZON,
        "context_hours": CONTEXT,
        "evaluation_window": "2023-08-01 00:00 through 2023-08-30 23:00; label starts [2928, 3649)",
        "raw_archive_rows_read_per_city": 3672,
        "raw_archive_cutoff": "2023-09-01 00:00 exclusive; no September outcome rows read",
        "archive": str(ARCHIVE),
        "archive_sha256": sha256(ARCHIVE),
        "prediction_sources": prediction_files,
        "models": list(MODEL_ORDER),
        "layers": ["city_total", "station_level"],
        "city_total_definition": "sum of all station-level hourly volume values within each city",
        "station_level_definition": "each charging station remains a separate series; fold statistics average over origins and stations",
        "seasonal_rules": {
            "Seasonal naive 24 h": "value at each forecast hour copied from the preceding 24-hour daily cycle",
            "Seasonal naive 168 h": "value at each forecast hour copied from the preceding 7-day weekly cycle",
            "causal_information": "only observed prefix rows before each origin are used",
        },
        "peak_level_definition": "max over the 24-hour forecast window; error is prediction peak minus actual peak",
        "peak_timing_definition": "first argmax hour within the 24-hour window; error is prediction index minus actual index",
        "ramp_definition": "absolute first difference between adjacent hourly values within the 24-hour forecast",
        "ramp_threshold_definition": f"city-total or station-specific q{int(RAMP_QUANTILE * 100)} of absolute hourly ramps in the observed prefix values[:{TRAIN_END}]",
        "ramp_threshold_is_test_tuned": False,
        "total_volume_definition": "sum across 24 forecast hours, evaluated at city-total and station level",
        "capacity_replay": {
            "capacity_definition": "max over 24-hour predicted city-total volume times (1 + safety margin)",
            "primary_safety_margin_pct": PRIMARY_MARGIN * 100.0,
            "safety_margin_grid_pct": [margin * 100.0 for margin in MARGINS],
            "primary_overload_penalty": PRIMARY_PENALTY,
            "overload_penalty_grid": list(PENALTIES),
            "overload_definition": "positive(actual city-total hourly volume - reserved capacity)",
            "raw_cost_proxy": "reserved capacity + lambda * overload volume",
            "normalized_cost_proxy": "raw cost proxy divided by realized 24-hour city-total peak for the same origin",
            "oracle_definition": "reserved capacity equals realized 24-hour city-total peak; unavailable lower-bound diagnostic",
        },
        "claim_limits": [
            "All operational metrics are retrospective diagnostics on the locked August evaluation window.",
            "Capacity values retain the CHARGED native aggregate-volume units; no kW conversion is assumed.",
            "Normalized cost is a dimensionless scenario proxy, not an electricity tariff or monetary cost.",
            "No claim is made about queueing, transformer loading, dispatch feasibility, grid reliability, or energy-market revenue.",
            "The oracle uses realized labels and is unavailable to a deployed forecaster; it is a lower-bound reference only.",
            "Fold is the independent unit for comparison; hourly origins and stations are nested repeated observations.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": [
            "operational_metrics_per_fold.csv",
            "operational_metrics_equal_fold.csv",
            "operational_metrics_per_origin_city_total.csv",
            "ramp_thresholds_by_city.csv",
            "capacity_reserve_risk_per_fold.csv",
            "capacity_reserve_risk_equal_fold.csv",
            "capacity_reserve_risk_per_origin.csv",
            "capacity_reserve_risk_primary_equal_fold.csv",
            "protocol.json",
            "summary.json",
            "README.md",
        ],
    }
    write_json(OUT / "protocol.json", manifest)

    summary = {
        "status": "completed",
        "fold_count": len(folds),
        "expected_fold_count": len(CITIES) * len(SEEDS),
        "origins_per_fold": N_ORIGINS,
        "operational_fold_rows": int(len(operational)),
        "operational_origin_rows_city_total": int(len(origins)),
        "capacity_fold_rows": int(len(capacities)),
        "capacity_origin_rows": int(len(capacity_origins)),
        "layers": ["city_total", "station_level"],
        "primary_capacity_rows": int(len(primary)),
        "primary_capacity_summary": primary_summary.to_dict(orient="records"),
        "equal_fold_operational_summary": equal_fold_operational.to_dict(orient="records"),
        "limitations": manifest["claim_limits"],
    }
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
    print(json.dumps({"status": result["status"], "fold_count": result["fold_count"], "capacity_fold_rows": result["capacity_fold_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute energy-facing metrics from frozen SCPRC prediction archives.

The script only reads existing NPZ files and writes a separate analysis run.
It keeps model lineage explicit: canonical arms, independent-switch controls,
operator controls, and Full-no-PCGrad are never silently merged.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = Path(__file__).resolve().parent
CITIES = ("ams", "jhb", "loa", "mel", "spo", "szh")
SEEDS = (13, 17, 29)
FOLDS = [(city, seed) for city in CITIES for seed in SEEDS]


def fold_name(city: str, seed: int) -> str:
    return f"target_{city}_seed{seed}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def operator_path(operator: str, city: str, seed: int) -> Path | None:
    root = ROOT / "experiments/supplementary/canonical_operator_controls_20261027" / operator
    candidates = sorted(root.glob(f"runs/{fold_name(city, seed)}/**/predictions.npz"))
    candidates += sorted(root.glob(f"{fold_name(city, seed)}/**/predictions.npz"))
    return candidates[0] if candidates else None


def model_specs(city: str, seed: int) -> dict[str, tuple[Path | None, str, str]]:
    fold = fold_name(city, seed)
    canonical = ROOT / 'experiments/forecasting/h24_framework' / fold / "predictions.npz"
    components = ROOT / 'experiments/independent_gates' / fold / "predictions.npz"
    original = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer" / fold / "predictions.npz"
    full_no_pcgrad = ROOT / "experiments/evaluation/scprc_mean_gradient" / fold / "predictions.npz"
    strict = OUT / "agent_strict_gap/runs" / fold / "predictions.npz"
    return {
        "original_phaseformer": (original, "pred_phaseformer_no_pcgrad_raw", "canonical_original"),
        "exact_pcgrad": (canonical, "pred_exact_pcgrad_raw", "canonical_four_arm"),
        "daily_relation": (canonical, "pred_daily_relation_pcgrad_raw", "canonical_four_arm"),
        "phase_relation": (canonical, "pred_phase_relation_pcgrad_raw", "canonical_four_arm"),
        "full_scpcr": (canonical, "pred_scpcr_candidate_raw", "canonical_four_arm"),
        "full_no_pcgrad": (full_no_pcgrad, "prediction_raw", "scprc_mean_gradient"),
        "scale_only_independent": (components, "pred_scale_only_raw", "p2_independent_switch"),
        "peer_only_independent": (components, "pred_peer_only_raw", "p2_independent_switch"),
        "source_disagreement_only_independent": (components, "pred_source_disagreement_only_raw", "p2_independent_switch"),
        "full_independent_switch": (components, "pred_full_raw", "p2_independent_switch"),
        "attention_operator": (operator_path("generic_attention", city, seed), "prediction_phasewise_attention_raw", "operator_control"),
        "deepsets_operator": (operator_path("deepsets", city, seed), "prediction_phaseformer_deepsets_adapter", "operator_control"),
        **{arm: (strict, f"pred_{arm}_raw", "strict_drop_20261104") for arm in (
            "full_drop_scale", "full_drop_peer", "full_drop_source_envelope", "daily_full_attenuation")},
    }


def metric_rows(truth: np.ndarray, prediction: np.ndarray, city: str, seed: int, model: str, lineage: str) -> tuple[dict, list[dict]]:
    truth = truth.astype(np.float64, copy=False)
    prediction = prediction.astype(np.float64, copy=False)
    city_truth = truth.sum(axis=2)
    city_prediction = prediction.sum(axis=2)
    city_error = city_prediction - city_truth
    total_truth = city_truth.sum(axis=1)
    total_prediction = city_prediction.sum(axis=1)
    total_error = total_prediction - total_truth
    peak_truth = city_truth.max(axis=1)
    peak_prediction = city_prediction.max(axis=1)
    peak_error = peak_prediction - peak_truth
    timing_error = city_prediction.argmax(axis=1) - city_truth.argmax(axis=1)
    ramp_truth = np.diff(city_truth, axis=1)
    ramp_prediction = np.diff(city_prediction, axis=1)
    ramp_error = ramp_prediction - ramp_truth
    cutoff = float(np.quantile(city_truth, 0.90))
    high_mask = city_truth >= cutoff
    site_error = prediction - truth
    positive_truth = np.maximum(truth, 0.0)
    metric = {
        "city": city.upper(),
        "seed": seed,
        "model": model,
        "lineage": lineage,
        "n_origins": int(truth.shape[0]),
        "horizon": int(truth.shape[1]),
        "n_sites": int(truth.shape[2]),
        "city_total_rmse": float(np.sqrt(np.mean(city_error ** 2))),
        "city_total_mae": float(np.mean(np.abs(city_error))),
        "peak_magnitude_mae": float(np.mean(np.abs(peak_error))),
        "peak_magnitude_rmse": float(np.sqrt(np.mean(peak_error ** 2))),
        "peak_signed_bias": float(np.mean(peak_error)),
        "peak_timing_mae_hours": float(np.mean(np.abs(timing_error))),
        "peak_timing_signed_bias_hours": float(np.mean(timing_error)),
        "ramp_mae": float(np.mean(np.abs(ramp_error))),
        "ramp_rmse": float(np.sqrt(np.mean(ramp_error ** 2))),
        "ramp_signed_bias": float(np.mean(ramp_error)),
        "total_24h_mae": float(np.mean(np.abs(total_error))),
        "total_24h_rmse": float(np.sqrt(np.mean(total_error ** 2))),
        "total_24h_signed_bias": float(np.mean(total_error)),
        "total_24h_relative_abs_error_pct": float(np.mean(np.abs(total_error) / np.maximum(np.abs(total_truth), 1e-12)) * 100),
        "high_load_city_mae": float(np.mean(np.abs(city_error[high_mask]))),
        "high_load_underforecast_mae": float(np.mean(np.maximum(-city_error[high_mask], 0.0))),
        "high_load_overforecast_mae": float(np.mean(np.maximum(city_error[high_mask], 0.0))),
        "load_weighted_site_mae": float(np.sum(positive_truth * np.abs(site_error)) / max(float(positive_truth.sum()), 1e-12)),
        "high_load_cutoff": cutoff,
        "high_load_definition": "realized city-total top decile within each fold",
    }
    lead_rows = []
    for lead in range(truth.shape[1]):
        err = city_error[:, lead]
        lead_rows.append({
            "city": city.upper(), "seed": seed, "model": model, "lineage": lineage,
            "lead": lead + 1, "city_total_rmse": float(np.sqrt(np.mean(err ** 2))),
            "city_total_mae": float(np.mean(np.abs(err))),
            "city_total_signed_bias": float(np.mean(err)),
        })
    return metric, lead_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


CORE_METRICS = ["peak_magnitude_mae", "ramp_mae", "total_24h_mae", "peak_timing_mae_hours"]
PAIRS = [
    ("original_phaseformer", "full_scpcr"), ("exact_pcgrad", "full_scpcr"),
    ("daily_relation", "full_scpcr"), ("phase_relation", "full_scpcr"),
    ("full_no_pcgrad", "full_scpcr"), ("attention_operator", "full_scpcr"),
    ("deepsets_operator", "full_scpcr"),
    ("scale_only_independent", "full_independent_switch"),
    ("peer_only_independent", "full_independent_switch"),
    ("source_disagreement_only_independent", "full_independent_switch"),
    ("full_drop_scale", "full_scpcr"), ("full_drop_peer", "full_scpcr"),
    ("full_drop_source_envelope", "full_scpcr"), ("daily_full_attenuation", "full_scpcr"),
]


def origin_scores(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    error = p - y
    peaks = p.max(axis=1) - y.max(axis=1)
    timing = p.argmax(axis=1) - y.argmax(axis=1)
    total = p.sum(axis=1) - y.sum(axis=1)
    ramps = np.diff(p, axis=1) - np.diff(y, axis=1)
    return {
        "peak_magnitude_mae": np.abs(peaks), "peak_signed_error": peaks,
        "ramp_mae": np.abs(ramps).mean(axis=1),
        "total_24h_mae": np.abs(total), "total_signed_error": total,
        "peak_timing_mae_hours": np.abs(timing), "peak_timing_signed_hours": timing,
        "actual_peak": y.max(axis=1), "predicted_peak": p.max(axis=1),
        "actual_peak_hour": y.argmax(axis=1), "predicted_peak_hour": p.argmax(axis=1),
        "actual_total": y.sum(axis=1), "predicted_total": p.sum(axis=1),
        "city_total_mae": np.abs(error).mean(axis=1),
        "city_total_mse": (error ** 2).mean(axis=1),
    }


def capacity_arrays(y: np.ndarray, p: np.ndarray, margin: float, penalty: float) -> dict[str, np.ndarray]:
    reserve = p.max(axis=1) * (1.0 + margin)
    actual_peak = y.max(axis=1)
    denominator = np.maximum(actual_peak, 1e-12)
    overload = np.maximum(y - reserve[:, None], 0)
    overload_volume = overload.sum(axis=1)
    excess = np.maximum(reserve - actual_peak, 0)
    return {
        "reserved_capacity": reserve, "overload_volume": overload_volume,
        "overload_penalty_component": penalty * overload_volume,
        "total_proxy_native": reserve + penalty * overload_volume,
        "reserve_normalized": reserve / denominator,
        "penalty_normalized": penalty * overload_volume / denominator,
        "total_proxy_normalized": (reserve + penalty * overload_volume) / denominator,
        "overload_window_rate_pct": 100.0 * (overload_volume > 0),
        "excess_reservation": excess,
        "excess_reservation_pct": 100.0 * excess / denominator,
        "normalization_denominator": denominator,
        "zero_denominator_protected": (actual_peak < 1e-12).astype(float),
    }


def evaluate_one_fold(city: str, seed: int) -> dict:
    specs = model_specs(city, seed)
    fold_dir = OUT / "01_metrics/fold_cache" / fold_name(city, seed)
    fold_dir.mkdir(parents=True, exist_ok=True)
    ref_path = specs["exact_pcgrad"][0]
    with np.load(ref_path, allow_pickle=False) as z:
        truth, starts, sites = z["truth_raw"], z["label_starts"], z["site_ids"]
    assert truth.ndim == 3 and truth.shape[:2] == (721, 24)
    assert len(sites) == truth.shape[2] and np.isfinite(truth).all()
    assert np.array_equal(starts, np.arange(2928, 3649))
    ref_hashes = tuple(map(sha256_array, (truth, starts, sites)))
    origins = pd.Timestamp("2023-04-01") + pd.to_timedelta(starts, unit="h")
    y = truth.sum(axis=2, dtype=np.float64)
    fold_rows, lead_rows, identity, capacity_rows, per_origin = [], [], [], [], []
    primitive_arrays, trajectories = {}, {"truth_city_total": y, "label_starts": starts}
    grouped: dict[Path, list[tuple[str, str, str]]] = {}
    for model, (path, key, lineage) in specs.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing archive for {city}/{seed}/{model}: {path}")
        grouped.setdefault(path, []).append((model, key, lineage))
    for path, models in grouped.items():
        file_hash = sha256_file(path)
        with np.load(path, allow_pickle=False) as z:
            these_hashes = tuple(sha256_array(z[k]) for k in ("truth_raw", "label_starts", "site_ids"))
            if these_hashes != ref_hashes:
                raise ValueError(f"Alignment mismatch {path}")
            for model, key, lineage in models:
                raw = z[key]
                if raw.shape != truth.shape or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid prediction {path}::{key}")
                identity.append({
                    "city": city.upper(), "seed": seed, "model": model, "lineage": lineage,
                    "path": str(path), "prediction_key": key, "status": "aligned",
                    "archive_sha256": file_hash, "prediction_array_sha256": sha256_array(raw),
                    "truth_sha256": ref_hashes[0], "label_starts_sha256": ref_hashes[1],
                    "site_ids_sha256": ref_hashes[2], "shape": str(raw.shape),
                    "negative_prediction_fraction": float((raw < 0).mean()),
                })
                for postprocess in ("raw", "nonnegative_clip"):
                    prediction = raw if postprocess == "raw" else np.maximum(raw, 0)
                    metric, leads = metric_rows(truth, prediction, city, seed, model, lineage)
                    metric["postprocess"] = postprocess
                    fold_rows.append(metric)
                    for lead in leads:
                        lead["postprocess"] = postprocess
                    lead_rows.extend(leads)
                    p = prediction.sum(axis=2, dtype=np.float64)
                    primitive = origin_scores(y, p)
                    primitive_arrays[f"{model}__{postprocess}"] = np.column_stack([primitive[m] for m in CORE_METRICS])
                    trajectories[f"{model}__{postprocess}"] = p
                    frame = pd.DataFrame(primitive)
                    frame.insert(0, "forecast_target_start", origins.strftime("%Y-%m-%d %H:%M:%S"))
                    frame.insert(0, "issue_time", (origins - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))
                    frame.insert(0, "label_start", starts)
                    frame.insert(0, "postprocess", postprocess)
                    frame.insert(0, "lineage", lineage)
                    frame.insert(0, "model", model)
                    frame.insert(0, "seed", seed)
                    frame.insert(0, "city", city.upper())
                    per_origin.append(frame)
                    if postprocess == "nonnegative_clip":
                        for margin in (0.0, 0.05, 0.10, 0.20):
                            for penalty in (1.0, 5.0, 10.0, 20.0):
                                arrays = capacity_arrays(y, p, margin, penalty)
                                capacity_rows.append({"city": city.upper(), "seed": seed, "model": model,
                                    "lineage": lineage, "postprocess": postprocess, "margin": margin,
                                    "penalty": penalty, **{k: float(v.mean()) for k, v in arrays.items()}})
                                if margin == .10 and penalty == 10:
                                    for k, v in arrays.items():
                                        frame[f"capacity_primary_{k}"] = v
                    del prediction
                del raw
    write_csv(fold_dir / "per_fold.csv", fold_rows)
    write_csv(fold_dir / "per_lead.csv", lead_rows)
    write_csv(fold_dir / "identity.csv", identity)
    write_csv(fold_dir / "capacity.csv", capacity_rows)
    pd.concat(per_origin, ignore_index=True).to_csv(fold_dir / "per_origin.csv.gz", index=False, compression="gzip")
    np.savez_compressed(fold_dir / "origin_primitives.npz", label_starts=starts, **primitive_arrays)
    np.savez_compressed(fold_dir / "city_trajectories.npz", **trajectories)
    print(f"EVALUATED {city.upper()} seed={seed} models={len(specs)} raw+clip", flush=True)
    return {"fold_rows": fold_rows, "lead_rows": lead_rows, "identity": identity, "capacity_rows": capacity_rows}


def summarize_complete(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = {"seed", "n_origins", "horizon", "n_sites"}
    numeric = [c for c in frame.select_dtypes(include="number").columns if c not in metadata]
    groups = ["city", "model", "lineage", "postprocess"]
    city = frame.groupby(groups)[numeric].mean().add_suffix("_mean").reset_index()
    city["folds"] = 3
    equal = frame.groupby(groups[1:])[numeric].mean().add_suffix("_mean").reset_index()
    equal.insert(0, "city", "EQUAL_FOLD_MEAN")
    equal["folds"] = 18
    return pd.concat([city, equal], ignore_index=True)


def write_uncertainty(frame: pd.DataFrame, block_reps: int = 2000) -> None:
    """Separate legacy fold bootstrap, city-cluster sensitivity, and time blocks.

    Folds share target truth and source training data; none of these intervals
    implies independence of cities or inference for new cities/retraining.
    """
    rng = np.random.default_rng(20261104)
    fold_draws = rng.integers(0, 18, size=(10000, 18))
    city_draws = rng.integers(0, 6, size=(10000, 6))
    result = []
    block_weights = []
    n, length = 721, 168
    for _ in CITIES:
        weight = np.zeros((block_reps, n), dtype=np.float64)
        for r in range(block_reps):
            selection = []
            for start in rng.integers(0, n - length + 1, size=int(np.ceil(n / length))):
                selection.extend(range(int(start), int(start) + length))
            weight[r] = np.bincount(selection[:n], minlength=n) / n
        block_weights.append(weight)
    origins = {}
    for city, seed in FOLDS:
        path = OUT / "01_metrics/fold_cache" / fold_name(city, seed) / "origin_primitives.npz"
        with np.load(path, allow_pickle=False) as z:
            origins[(city.upper(), seed)] = {k: z[k] for k in z.files if k != "label_starts"}
    for postprocess in ("raw", "nonnegative_clip"):
        sub = frame[frame.postprocess == postprocess].set_index(["city", "seed", "model"])
        keys = [(c.upper(), s) for c, s in FOLDS]
        for left, right in PAIRS:
            a = np.asarray([[sub.loc[(*fold, left), m] for m in CORE_METRICS] for fold in keys])
            b = np.asarray([[sub.loc[(*fold, right), m] for m in CORE_METRICS] for fold in keys])
            delta = a - b
            schemes = {
                "paired_18_fold_bootstrap_conditional": delta[fold_draws].mean(axis=1),
                "six_city_cluster_bootstrap_all_seeds": delta.reshape(6, 3, 4).mean(axis=1)[city_draws].mean(axis=1),
            }
            draws = np.zeros((block_reps, len(CORE_METRICS)))
            for c_idx, city in enumerate(CITIES):
                for seed in SEEDS:
                    primitive = origins[(city.upper(), seed)]
                    difference = primitive[f"{left}__{postprocess}"] - primitive[f"{right}__{postprocess}"]
                    draws += block_weights[c_idx] @ difference / 18
            schemes["paired_168_origin_moving_block_fixed_cities_seeds"] = draws
            relative = 100 * (a - b) / np.maximum(np.abs(a), 1e-12)
            for method, samples in schemes.items():
                for j, metric in enumerate(CORE_METRICS):
                    result.append({"left_model": left, "right_model": right, "postprocess": postprocess,
                        "metric": metric, "mean_difference": float(delta[:, j].mean()),
                        "difference_definition": "left error minus right error; positive favors right",
                        "ratio_of_equal_fold_mean_gain_pct_for_right": float(100 * (a[:, j].mean() - b[:, j].mean()) / a[:, j].mean()),
                        "mean_fold_relative_gain_pct_for_right": float(relative[:, j].mean()),
                        "ci95_low": float(np.quantile(samples[:, j], .025)),
                        "ci95_high": float(np.quantile(samples[:, j], .975)),
                        "interval_method": method, "n_cities": 6, "n_seeds": 3, "n_folds": 18,
                        "right_better_folds": int((delta[:, j] > 0).sum()),
                        "right_better_cities": int((delta[:, j].reshape(6, 3).mean(axis=1) > 0).sum()),
                        "bootstrap_replicates": len(samples), "seed": 20261104,
                        "multiplicity": "exploratory pointwise intervals; no familywise claim"})
    pd.DataFrame(result).to_csv(OUT / "02_paired_stats/paired_energy_uncertainty.csv", index=False)
    interaction_rows = []
    for postprocess in ("raw", "nonnegative_clip"):
        sub = frame[frame.postprocess == postprocess].set_index(["city", "seed", "model"])
        for metric in CORE_METRICS:
            values = np.asarray([sub.loc[(c.upper(), s, "full_scpcr"), metric] - sub.loc[(c.upper(), s, "exact_pcgrad"), metric]
                - sub.loc[(c.upper(), s, "full_no_pcgrad"), metric] + sub.loc[(c.upper(), s, "original_phaseformer"), metric]
                for c, s in FOLDS])
            samples = values.reshape(6, 3).mean(axis=1)[city_draws].mean(axis=1)
            interaction_rows.append({"postprocess": postprocess, "metric": metric,
                "definition": "(Full-Exact)-(Full_no_PCGrad-Original); negative means larger error reduction with PCGrad",
                "interaction": values.mean(), "city_cluster_ci95_low": np.quantile(samples, .025),
                "city_cluster_ci95_high": np.quantile(samples, .975), "cities": 6, "folds": 18})
    pd.DataFrame(interaction_rows).to_csv(OUT / "02_paired_stats/factorial_energy_interaction.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    for subdir in ("00_identity", "01_metrics", "02_paired_stats", "04_capacity", "06_status"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one_fold, city, seed) for city, seed in FOLDS]
            for future in as_completed(futures):
                future.result()
    def collect(filename: str) -> pd.DataFrame:
        return pd.concat([pd.read_csv(OUT / "01_metrics/fold_cache" / fold_name(c, s) / filename) for c, s in FOLDS], ignore_index=True)
    folds, identity, lead, capacity = map(collect, ("per_fold.csv", "identity.csv", "per_lead.csv", "capacity.csv"))
    assert len(folds) == 576 and len(identity) == 288 and (identity.status == "aligned").all()
    assert not folds.duplicated(["city", "seed", "model", "postprocess"]).any()
    folds.to_csv(OUT / "01_metrics/per_fold_energy_metrics.csv", index=False)
    identity.to_csv(OUT / "00_identity/archive_identity.csv", index=False)
    pd.DataFrame(columns=["city", "seed", "model", "status"]).to_csv(OUT / "00_identity/missing_or_mismatched.csv", index=False)
    lead.to_csv(OUT / "01_metrics/per_lead_city_total_metrics.csv", index=False)
    summarize_complete(folds).to_csv(OUT / "01_metrics/city_and_equal_fold_summary.csv", index=False)
    capacity.to_csv(OUT / "04_capacity/capacity_per_fold_sensitivity.csv", index=False)
    numeric = [c for c in capacity.select_dtypes(include="number").columns if c not in {"seed", "margin", "penalty"}]
    capacity.groupby(["model", "postprocess", "margin", "penalty"])[numeric].mean().reset_index().to_csv(OUT / "04_capacity/capacity_equal_fold_sensitivity.csv", index=False)
    with gzip.open(OUT / "01_metrics/per_origin_energy_metrics.csv.gz", "wt", encoding="utf-8") as stream:
        for i, (city, seed) in enumerate(FOLDS):
            frame = pd.read_csv(OUT / "01_metrics/fold_cache" / fold_name(city, seed) / "per_origin.csv.gz")
            frame.to_csv(stream, index=False, header=(i == 0))
    print("Fold tables and per-origin export complete; computing paired uncertainty", flush=True)
    write_uncertainty(folds)
    status = {"run_id": "SCPRC-ENERGY-VALUE-20261104", "evaluation_completed": True,
        "strict_training_completed": True, "strict_arm_trainings": 72, "folds": 18, "model_identities": 16,
        "prediction_postprocesses": ["raw", "nonnegative_clip"], "fold_metric_rows": len(folds),
        "aligned_model_fold_rows": len(identity), "source_archives_modified": False,
        "primary_point_protocol": "raw unchanged", "operation_protocol": "max(raw,0) unchanged",
        "retrospective": True, "blind_confirmation": False,
        "uncertainty": "three explicitly distinct conditional sensitivity schemes; no familywise significance claim"}
    (OUT / "06_status/run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()

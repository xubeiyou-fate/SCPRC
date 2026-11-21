#!/usr/bin/env python3
"""Paired moving-block intervals for external forecast-error differences."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import experiments.commands.summarize_external_baseline_results as audit

OUT = audit.OUT
SEEDS = audit.SEEDS
DATASETS = audit.DATASETS
BLOCK_LENGTH = 168
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261028
CANDIDATE = "SCPRC candidate (v2.2 legacy external matrix)"
COMPARISONS = (
    "PhaseFormer Exact (legacy external matrix)",
    "Persistence / last-value rule (legacy external matrix)",
    "Daily-mean rule (legacy external matrix)",
    "Weekly-naive rule (legacy external matrix)",
    "DLINEAR",
    "PATCHTST",
    "TIMEMIXER",
    "NBEATS_GRAPH",
)


def model_errors(dataset: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truths, sq_errors, abs_errors = [], [], []
    for seed in SEEDS:
        old_truth, old_predictions = audit.existing_arrays(dataset, seed)
        if model in old_predictions:
            prediction = old_predictions[model]
            truth = old_truth
        else:
            method = model.lower()
            truth, prediction, _ = audit.new_predictions(dataset, method, seed)
            if not np.allclose(old_truth, truth, rtol=0.0, atol=1e-6):
                raise ValueError(f"target truth mismatch: {dataset} seed={seed} model={model}")
        error = prediction.astype(np.float64) - truth.astype(np.float64)
        truths.append(truth.shape)
        sq_errors.append(np.square(error).sum(axis=(1, 2)))
        abs_errors.append(np.abs(error).sum(axis=(1, 2)))
    if len(set(truths)) != 1:
        raise ValueError(f"inconsistent truth shapes for {dataset} {model}: {truths}")
    return np.stack(sq_errors), np.stack(abs_errors), np.asarray(truths[0], dtype=np.int64)


def moving_block_indices(origins: int, rng: np.random.Generator) -> np.ndarray:
    if origins <= BLOCK_LENGTH:
        return np.arange(origins)
    block_count = int(np.ceil(origins / BLOCK_LENGTH))
    starts = rng.integers(0, origins - BLOCK_LENGTH + 1, size=block_count)
    indices = np.concatenate([np.arange(start, start + BLOCK_LENGTH) for start in starts])
    return indices[:origins]


def paired_interval(candidate_sq: np.ndarray, candidate_abs: np.ndarray,
                    baseline_sq: np.ndarray, baseline_abs: np.ndarray,
                    shape: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    seeds, origins = candidate_sq.shape
    horizons, sites = int(shape[1]), int(shape[2])
    divisor = seeds * origins * horizons * sites
    candidate_rmse = float(np.sqrt(candidate_sq.sum() / divisor))
    baseline_rmse = float(np.sqrt(baseline_sq.sum() / divisor))
    candidate_mae = float(candidate_abs.sum() / divisor)
    baseline_mae = float(baseline_abs.sum() / divisor)
    rmse_diff, mae_diff = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = moving_block_indices(origins, rng)
        count = seeds * len(indices) * horizons * sites
        cand_rmse = np.sqrt(candidate_sq[:, indices].sum() / count)
        base_rmse = np.sqrt(baseline_sq[:, indices].sum() / count)
        cand_mae = candidate_abs[:, indices].sum() / count
        base_mae = baseline_abs[:, indices].sum() / count
        rmse_diff.append(float(base_rmse - cand_rmse))
        mae_diff.append(float(base_mae - cand_mae))
    rmse_low, rmse_high = np.quantile(rmse_diff, [0.025, 0.975])
    mae_low, mae_high = np.quantile(mae_diff, [0.025, 0.975])
    return {
        "candidate_rmse": candidate_rmse,
        "baseline_rmse": baseline_rmse,
        "baseline_minus_candidate_rmse": baseline_rmse - candidate_rmse,
        "rmse_diff_ci95_low": float(rmse_low),
        "rmse_diff_ci95_high": float(rmse_high),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "baseline_minus_candidate_mae": baseline_mae - candidate_mae,
        "mae_diff_ci95_low": float(mae_low),
        "mae_diff_ci95_high": float(mae_high),
    }


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for dataset in DATASETS:
        cand_sq, cand_abs, shape = model_errors(dataset, CANDIDATE)
        for baseline in COMPARISONS:
            base_sq, base_abs, baseline_shape = model_errors(dataset, baseline)
            if not np.array_equal(shape, baseline_shape):
                raise ValueError(f"shape mismatch for {dataset} {baseline}: {shape} vs {baseline_shape}")
            rows.append({
                "dataset": dataset,
                "candidate": CANDIDATE,
                "baseline": baseline,
                "origins": int(shape[0]),
                "horizon": int(shape[1]),
                "sites": int(shape[2]),
                "training_seeds_fixed": ";".join(map(str, SEEDS)),
                "block_length_origins": BLOCK_LENGTH,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "interval_scope": "temporal moving-block uncertainty conditional on three fixed trained seeds; not population uncertainty over cities",
                **paired_interval(cand_sq, cand_abs, base_sq, base_abs, shape, rng),
            })
    output = OUT / "paired_block_bootstrap_168origin.csv"
    columns = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired intervals to {output}")


if __name__ == "__main__":
    main()

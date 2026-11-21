#!/usr/bin/env python3
"""Canonical Original-versus-Full fold-level paired uncertainty summary."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = Path(__file__).resolve().parent
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
FULL_ROOT = ROOT / 'experiments/forecasting/h24_framework'
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
REPLICATES = 10000
RNG_SEED = 20261031


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    site_mse = np.mean(error * error, axis=(0, 1))
    site_mae = np.mean(np.abs(error), axis=(0, 1))
    return {
        "global_rmse": float(np.sqrt(np.mean(error * error))),
        "global_mae": float(np.mean(np.abs(error))),
        "macro_site_rmse": float(np.mean(np.sqrt(site_mse))),
        "macro_site_mae": float(np.mean(site_mae)),
    }


def bootstrap(rows: list[dict], metric: str, rng: np.random.Generator) -> dict[str, float]:
    original = np.asarray([row[f"original_{metric}"] for row in rows], dtype=np.float64)
    full = np.asarray([row[f"full_{metric}"] for row in rows], dtype=np.float64)
    n = len(rows)
    sampled = rng.integers(0, n, size=(REPLICATES, n))
    original_means = original[sampled].mean(axis=1)
    full_means = full[sampled].mean(axis=1)
    deltas = full_means - original_means
    relative = 100.0 * (full_means / np.maximum(original_means, 1e-12) - 1.0)
    return {
        "original_point": float(original.mean()),
        "full_point": float(full.mean()),
        "absolute_delta_point": float(full.mean() - original.mean()),
        "relative_delta_point_percent": float(100.0 * (full.mean() / max(original.mean(), 1e-12) - 1.0)),
        "absolute_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "absolute_delta_ci95_high": float(np.quantile(deltas, 0.975)),
        "relative_delta_ci95_low_percent": float(np.quantile(relative, 0.025)),
        "relative_delta_ci95_high_percent": float(np.quantile(relative, 0.975)),
        "better_folds": int(np.sum(full < original)),
        "folds": n,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_unit": "city x seed fold; paired resampling of the 18 complete fold metrics",
    }


def main() -> None:
    rows = []
    input_hashes = {}
    for city in CITIES:
        for seed in SEEDS:
            key = f"target_{city.lower()}_seed{seed}"
            original_path = ORIGINAL_ROOT / key / "predictions.npz"
            full_path = FULL_ROOT / key / "predictions.npz"
            with np.load(original_path, allow_pickle=False) as original, np.load(full_path, allow_pickle=False) as full:
                truth = original["truth_raw"]
                checks = {
                    "truth": np.array_equal(truth, full["truth_raw"]),
                    "origins": np.array_equal(original["label_starts"], full["label_starts"]),
                    "site_ids": np.array_equal(original["site_ids"], full["site_ids"]),
                }
                if not all(checks.values()):
                    raise RuntimeError(f"lineage mismatch in {key}: {checks}")
                original_prediction = original["pred_phaseformer_no_pcgrad_raw"]
                full_prediction = full["pred_scpcr_candidate_raw"]
            original_score = score(original_prediction, truth)
            full_score = score(full_prediction, truth)
            row = {"city": city, "seed": seed, "origins": int(truth.shape[0]), "horizon": int(truth.shape[1]), "sites": int(truth.shape[2]), "alignment_passed": True}
            for metric in ("global_rmse", "global_mae", "macro_site_rmse", "macro_site_mae"):
                row[f"original_{metric}"] = original_score[metric]
                row[f"full_{metric}"] = full_score[metric]
                row[f"full_minus_original_{metric}"] = full_score[metric] - original_score[metric]
                row[f"full_vs_original_{metric}_relative_percent"] = 100.0 * (full_score[metric] / max(original_score[metric], 1e-12) - 1.0)
            rows.append(row)
            input_hashes[str(original_path.relative_to(ROOT))] = sha256(original_path)
            input_hashes[str(full_path.relative_to(ROOT))] = sha256(full_path)
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 folds, found {len(rows)}")
    per_fold = OUT / "paired_fold_metrics.csv"
    with per_fold.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(RNG_SEED)
    summary = {}
    for metric in ("global_rmse", "global_mae", "macro_site_rmse", "macro_site_mae"):
        summary[metric] = bootstrap(rows, metric, rng)
    payload = {
        "status": "completed",
        "experiment": "paired fold-level uncertainty: canonical Original PhaseFormer vs Full SCPRC",
        "retrospective": True,
        "lineage_check": "lineage_check.json",
        "protocol_id": "scpcr-matched-baseline-20261023",
        "context_hours": 168,
        "horizon_hours": 24,
        "fold_count": 18,
        "cities": list(CITIES),
        "seeds": list(SEEDS),
        "primary_estimand": "equal-weight mean of 18 city-seed fold metrics",
        "uncertainty": {
            "method": "paired fold bootstrap with identical resampled fold indices for Original and Full",
            "unit": "city x seed fold",
            "replicates": REPLICATES,
            "rng_seed": RNG_SEED,
            "scope": "conditional on six observed cities, three fixed seeds, frozen checkpoints, and fixed retrospective origins",
            "not_claimed": ["unseen-city generalization uncertainty", "retraining uncertainty", "prospective confirmation", "IID origin-level uncertainty"],
        },
        "summary": summary,
        "input_hashes": input_hashes,
        "old_results_overwritten": False,
    }
    (OUT / "paired_fold_ci.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Paired fold-level uncertainty: Original PhaseFormer vs Full SCPRC",
        "",
        "Status: **completed**; 18 city×seed folds; 10,000 paired fold bootstrap replicates.",
        "",
        "The primary estimator is the equal-weight mean of the 18 fold-level metrics. Each bootstrap draw resamples complete city×seed folds and uses the same indices for Original and Full. The intervals condition on the six observed cities, three fixed seeds, frozen checkpoints, and the retrospective forecast origins.",
        "",
        "| Metric | Original | Full | Full−Original | 95% CI for difference | Relative change | 95% CI relative | Better folds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"global_rmse": "Global RMSE", "global_mae": "Global MAE", "macro_site_rmse": "Macro-site RMSE", "macro_site_mae": "Macro-site MAE"}
    for metric in ("global_rmse", "global_mae", "macro_site_rmse", "macro_site_mae"):
        item = summary[metric]
        lines.append(f"| {labels[metric]} | {item['original_point']:.6f} | {item['full_point']:.6f} | {item['absolute_delta_point']:+.6f} | [{item['absolute_delta_ci95_low']:+.6f}, {item['absolute_delta_ci95_high']:+.6f}] | {item['relative_delta_point_percent']:+.3f}% | [{item['relative_delta_ci95_low_percent']:+.3f}%, {item['relative_delta_ci95_high_percent']:+.3f}%] | {item['better_folds']}/{item['folds']} |")
    lines += [
        "",
        "Negative differences favor Full SCPRC. These are fold-level uncertainty intervals, not guarantees over unseen cities, retraining runs, or future prospective data.",
        "",
        "Input lineage was checked against `lineage_check.json`; per-fold values are in `paired_fold_metrics.csv`.",
    ]
    (OUT / "paired_fold_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "folds": len(rows), "metrics": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

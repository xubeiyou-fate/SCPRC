#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CANONICAL = ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
BLOCK_LENGTH = 168
BOOTSTRAP_REPLICATES = 5000
RNG_SEED = 20261027
ARMS = ("exact", "daily", "phase", "scpcr", "attention", "deepsets")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def score(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(error * error))), float(np.mean(np.abs(error)))


def origin_sums(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return np.sum(error * error, axis=(1, 2)), np.sum(np.abs(error), axis=(1, 2))


def block_sums(values: np.ndarray, length: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return cumulative[length:] - cumulative[:-length]


def prepare_bootstrap(sums: dict[str, tuple[np.ndarray, np.ndarray]], sites: int) -> dict[str, Any]:
    origins = len(next(iter(sums.values()))[0])
    lengths = [BLOCK_LENGTH] * (origins // BLOCK_LENGTH)
    if origins % BLOCK_LENGTH:
        lengths.append(origins % BLOCK_LENGTH)
    prepared: dict[str, Any] = {"sites": sites, "lengths": lengths, "blocks": {}}
    for arm in ARMS:
        sse, sae = sums[arm]
        prepared["blocks"][arm] = {
            "sse": [block_sums(sse, length) for length in lengths],
            "sae": [block_sums(sae, length) for length in lengths],
        }
    return prepared


def paired_bootstrap(folds: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    comparisons = {
        "attention_vs_scpcr": {"rmse": [], "mae": []},
        "deepsets_vs_scpcr": {"rmse": [], "mae": []},
        "scpcr_vs_exact": {"rmse": [], "mae": []},
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        totals = {arm: {"sse": [], "sae": []} for arm in ARMS}
        for fold in folds:
            sampled = [rng.integers(0, len(fold["blocks"]["scpcr"]["sse"][i]))
                       for i in range(len(fold["lengths"]))]
            count = sum(fold["lengths"]) * 24 * fold["sites"]
            for arm in ARMS:
                parts = fold["blocks"][arm]
                sse = sum(float(parts["sse"][i][sampled[i]]) for i in range(len(sampled)))
                sae = sum(float(parts["sae"][i][sampled[i]]) for i in range(len(sampled)))
                totals[arm]["sse"].append(np.sqrt(sse / count))
                totals[arm]["sae"].append(sae / count)
        means = {
            arm: {
                "rmse": float(np.mean(arm_totals["sse"])),
                "mae": float(np.mean(arm_totals["sae"])),
            }
            for arm, arm_totals in totals.items()
        }
        for name, challenger, reference in (
            ("attention_vs_scpcr", "attention", "scpcr"),
            ("deepsets_vs_scpcr", "deepsets", "scpcr"),
            ("scpcr_vs_exact", "scpcr", "exact"),
        ):
            for metric in ("rmse", "mae"):
                comparisons[name][metric].append(100.0 * (means[challenger][metric] / means[reference][metric] - 1.0))
    return {
        name: {
            metric: {
                "ci95_percentile": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
                "bootstrap_mean_percent": float(np.mean(samples)),
                "replicates": BOOTSTRAP_REPLICATES,
                "block_length_origins": BLOCK_LENGTH,
                "scope": "temporal-origin uncertainty conditional on the 18 fixed city-seed folds",
            }
            for metric, samples in values.items()
        }
        for name, values in comparisons.items()
    }


def prediction_path(model: str, city: str, seed: int) -> Path:
    if model == "generic_attention":
        return OUT / model / f"target_{city.lower()}_seed{seed}" / "predictions.npz"
    return OUT / model / "runs" / f"target_{city.lower()}_seed{seed}" / "attempt_01" / "predictions.npz"


def result_path(model: str, city: str, seed: int) -> Path:
    return prediction_path(model, city, seed).with_name("result.json")


def main() -> None:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("completed_total_folds") != 36:
        raise RuntimeError("operator matrix run is not complete; refusing to publish audit")
    rows = []
    bootstrap_folds = []
    input_hashes = {}
    for city in CITIES:
        for seed in SEEDS:
            fold_key = f"target_{city.lower()}_seed{seed}"
            canonical_path = CANONICAL / fold_key / "predictions.npz"
            attention_path = prediction_path("generic_attention", city, seed)
            deepsets_path = prediction_path("deepsets", city, seed)
            with np.load(canonical_path, allow_pickle=False) as canonical:
                truth = canonical["truth_raw"]
                starts = canonical["label_starts"]
                sites = canonical["site_ids"]
                predictions = {
                    "exact": canonical["pred_exact_pcgrad_raw"],
                    "daily": canonical["pred_daily_relation_pcgrad_raw"],
                    "phase": canonical["pred_phase_relation_pcgrad_raw"],
                    "scpcr": canonical["pred_scpcr_candidate_raw"],
                }
            with np.load(attention_path, allow_pickle=False) as attention:
                checks = {
                    "attention_truth": np.array_equal(attention["truth_raw"], truth),
                    "attention_origins": np.array_equal(attention["label_starts"], starts),
                    "attention_sites": np.array_equal(attention["site_ids"], sites),
                }
                predictions["attention"] = attention["prediction_phasewise_attention_raw"]
            with np.load(deepsets_path, allow_pickle=False) as deepsets:
                deepsets_prediction_keys = [key for key in deepsets.files if key.startswith("prediction_")]
                if len(deepsets_prediction_keys) != 1:
                    raise RuntimeError(f"unexpected DeepSets prediction keys in {deepsets_path}: {deepsets_prediction_keys}")
                checks.update({
                    "deepsets_truth": np.array_equal(deepsets["truth_raw"], truth),
                    "deepsets_origins": np.array_equal(deepsets["label_starts"], starts),
                    "deepsets_sites": np.array_equal(deepsets["site_ids"], sites),
                })
                predictions["deepsets"] = deepsets[deepsets_prediction_keys[0]]
            if not all(checks.values()):
                raise RuntimeError(f"prediction matrix alignment failed for {city}/seed{seed}: {checks}")
            fold_metrics = {arm: score(predictions[arm], truth) for arm in ARMS}
            for model, artifact in (("generic_attention", attention_path), ("deepsets", deepsets_path)):
                result = json.loads(result_path(model, city, seed).read_text(encoding="utf-8"))
                stored = result["metrics"]
                actual = fold_metrics["attention" if model == "generic_attention" else "deepsets"]
                if not np.allclose([stored["rmse"], stored["mae"]], actual, rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"stored-vs-recomputed metric mismatch: {result_path(model, city, seed)}")
                input_hashes[str(artifact.relative_to(ROOT))] = sha256(artifact)
                result_file = result_path(model, city, seed)
                input_hashes[str(result_file.relative_to(ROOT))] = sha256(result_file)
            input_hashes[str(canonical_path.relative_to(ROOT))] = sha256(canonical_path)
            row = {"city": city, "seed": seed, "origins": int(truth.shape[0]), "sites": int(truth.shape[2]),
                   "alignment_passed": True}
            for arm, (rmse, mae) in fold_metrics.items():
                row[f"{arm}_rmse"] = rmse
                row[f"{arm}_mae"] = mae
            for challenger in ("attention", "deepsets", "scpcr"):
                reference = "scpcr" if challenger != "scpcr" else "exact"
                for metric_index, metric in enumerate(("rmse", "mae")):
                    row[f"{challenger}_vs_{reference}_{metric}_pct"] = 100.0 * (
                        fold_metrics[challenger][metric_index] / fold_metrics[reference][metric_index] - 1.0
                    )
            rows.append(row)
            sums = {arm: origin_sums(predictions[arm], truth) for arm in ARMS}
            bootstrap_folds.append(prepare_bootstrap(sums, int(truth.shape[2])))
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 aligned folds, found {len(rows)}")
    means = {
        arm: {metric: float(np.mean([row[f"{arm}_{metric}"] for row in rows])) for metric in ("rmse", "mae")}
        for arm in ARMS
    }
    comparisons = {}
    for name, challenger, reference in (
        ("attention_vs_scpcr", "attention", "scpcr"),
        ("deepsets_vs_scpcr", "deepsets", "scpcr"),
        ("scpcr_vs_exact", "scpcr", "exact"),
    ):
        comparisons[name] = {}
        for metric in ("rmse", "mae"):
            fold_key = f"{challenger}_vs_{reference}_{metric}_pct"
            comparisons[name][metric] = {
                "ratio_of_equal_fold_mean_metrics_percent": 100.0 * (means[challenger][metric] / means[reference][metric] - 1.0),
                "mean_fold_relative_change_percent": float(np.mean([row[fold_key] for row in rows])),
                "folds_better": int(sum(row[f"{challenger}_{metric}"] < row[f"{reference}_{metric}"] for row in rows)),
                "fold_count": len(rows),
            }
    summary = {
        "status": "completed_audited_retrospective_operator_comparison",
        "protocol_id": "scpcr-matched-baseline-20261023",
        "protocol_sha256": PROTOCOL_SHA256,
        "retrospective": True,
        "blind_confirmation": False,
        "fold_count": len(rows),
        "origins_per_fold": 721,
        "aggregation": "equal-weight mean of fold-level raw-volume RMSE and MAE",
        "models": means,
        "comparisons": comparisons,
        "paired_temporal_block_bootstrap": paired_bootstrap(bootstrap_folds),
        "bootstrap_scope": "168-origin moving blocks conditional on six fixed cities and three fixed seeds; not city-population, retraining, or prospective uncertainty",
        "alignment": "truth arrays, site IDs, and forecast-origin arrays exactly match the canonical four-arm matrix in every fold",
        "metric_recomputation": "attention and DeepSets fold metrics match their stored result.json values within 1e-6 relative/absolute tolerance",
        "input_sha256": input_hashes,
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "audited_at_unix": time.time(),
    }
    write_json(OUT / "audit_summary.json", summary)
    with (OUT / "per_fold_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {"exact": "Exact-PCGrad", "daily": "Daily relation-PCGrad", "phase": "Phase relation-PCGrad",
              "scpcr": "Full SCPRC", "attention": "Generic attention", "deepsets": "DeepSets adapter"}
    lines = [
        "# Audited canonical relation-operator comparison (2026-10-27)",
        "",
        "Status: **completed; retrospective, not blind confirmation**.",
        "",
        f"Protocol: `scpcr-matched-baseline-20261023`; SHA-256 `{PROTOCOL_SHA256}`.",
        "All six arms have 18 city-seed folds; attention and DeepSets predictions are exactly aligned to the canonical four-arm matrix on truth, site IDs, and forecast origins.",
        "The primary estimator is the equal-fold mean of raw-volume RMSE/MAE. Bootstrap intervals quantify only temporal-origin uncertainty conditional on the six cities and three fixed seeds.",
        "",
        "| Model | Folds | Equal-fold RMSE | Equal-fold MAE |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(f"| {labels[arm]} | 18 | {means[arm]['rmse']:.4f} | {means[arm]['mae']:.4f} |")
    lines.extend(["", "| Comparison (challenger vs reference) | RMSE ratio-of-means | MAE ratio-of-means | Better folds RMSE / MAE |", "|---|---:|---:|---:|"])
    for name, challenger, reference in (
        ("Generic attention vs Full SCPRC", "attention_vs_scpcr", "attention"),
        ("DeepSets adapter vs Full SCPRC", "deepsets_vs_scpcr", "deepsets"),
        ("Full SCPRC vs Exact-PCGrad", "scpcr_vs_exact", "scpcr"),
    ):
        comp = comparisons[challenger]
        label = f"{name}"
        lines.append(f"| {label} | {comp['rmse']['ratio_of_equal_fold_mean_metrics_percent']:+.2f}% | {comp['mae']['ratio_of_equal_fold_mean_metrics_percent']:+.2f}% | {comp['rmse']['folds_better']}/18 / {comp['mae']['folds_better']}/18 |")
    lines.extend([
        "",
        "Paired 95% temporal-block intervals:",
        "",
    ])
    for comp_name in ("attention_vs_scpcr", "deepsets_vs_scpcr", "scpcr_vs_exact"):
        for metric in ("rmse", "mae"):
            interval = summary["paired_temporal_block_bootstrap"][comp_name][metric]["ci95_percentile"]
            lines.append(f"- `{comp_name}` {metric.upper()}: {interval[0]:+.2f}% to {interval[1]:+.2f}%.")
    lines.extend([
        "",
        "This is a finite retrospective comparison of the tested implementations, not universal operator superiority, independent-city generalization, retraining uncertainty, or a prospective guarantee.",
        "",
        "Artifacts: `audit_summary.json`, `per_fold_audit.csv`; training records remain in the per-fold directories.",
    ])
    (OUT / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

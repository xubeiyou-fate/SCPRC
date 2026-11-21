#!/usr/bin/env python3
"""Recompute and summarize three external source-only baseline matrices."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = ROOT / "experiments/supplementary/external_strong_baselines_20261028"
EXISTING = ROOT / "experiments/supplementary/external_current_20261024"
URBAN_CSV = ROOT / "research/urbanev_timemixer/sources/UrbanEV/data/volume.csv"
DATASETS = ("jiaxing", "shanghai", "urbanev")
SEEDS = (13, 17, 29)


def score(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if truth.shape != prediction.shape:
        raise ValueError(f"truth/prediction shape mismatch: {truth.shape} != {prediction.shape}")
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error)))


def existing_arrays(dataset: str, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset == "jiaxing":
        path = EXISTING / dataset / f"seed_{seed}" / "predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            truth = archive["truth_kwh"]
            return truth, {
                "PhaseFormer Exact (legacy external matrix)": archive["pred_phaseformer_exact"],
                "SCPRC candidate (v2.2 legacy external matrix)": archive["pred_scpcr_v22_pcgrad_candidate_diagnostic"],
                "Persistence / last-value rule (legacy external matrix)": archive["pred_persistence"],
                "Daily-mean rule (legacy external matrix)": archive["pred_daily_mean"],
                "Weekly-naive rule (legacy external matrix)": archive["pred_weekly_naive"],
            }
    if dataset == "shanghai":
        path = EXISTING / dataset / f"seed{seed}" / "q4_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            truth = archive["truth_kw"]
            return truth, {
                "PhaseFormer Exact (legacy external matrix)": archive["pred_phaseformer_exact_kw"],
                "SCPRC candidate (v2.2 legacy external matrix)": archive["pred_scpcr_candidate_kw"],
                "Persistence / last-value rule (legacy external matrix)": archive["pred_persistence_kw"],
                "Daily-mean rule (legacy external matrix)": archive["pred_daily_mean_7d_kw"],
                "Weekly-naive rule (legacy external matrix)": archive["pred_weekly_naive_kw"],
            }
    result_path = EXISTING / dataset / f"seed_{seed}" / "result.json"
    prediction_path = EXISTING / dataset / f"seed_{seed}" / "predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        starts = archive["february_starts"].astype(np.int64)
        site_ids = tuple(map(str, archive["site_ids"]))
        raw = pd.read_csv(URBAN_CSV)
        if tuple(map(str, raw.columns[1:])) != site_ids:
            raise ValueError("UrbanEV site ID order differs between prediction cache and volume.csv")
        values = raw.iloc[:, 1:].to_numpy(dtype=np.float32)
        truth = np.stack([values[start:start + 24] for start in starts])
        predictions = {
            "PhaseFormer Exact (legacy external matrix)": archive["february__phaseformer_exact_source_scale"],
            "SCPRC candidate (v2.2 legacy external matrix)": archive["february__scpcr_v22_pcgrad_candidate_diagnostic_source_scale"],
            "Persistence / last-value rule (legacy external matrix)": archive["february__last_value"],
            "Daily-mean rule (legacy external matrix)": archive["february__daily_mean"],
            "Weekly-naive rule (legacy external matrix)": archive["february__weekly_naive"],
        }
    if result.get("seed") != seed or result.get("status") != "completed":
        raise ValueError(f"invalid archived UrbanEV run metadata: {result_path}")
    return truth, predictions


def new_predictions(dataset: str, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    run_dir = OUT / method / dataset / f"seed_{seed}"
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing completed {method}/{dataset}/seed_{seed} artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or int(result.get("seed", -1)) != seed:
        raise ValueError(f"incomplete or mismatched run: {result_path}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if method == "patchtst":
            prediction = archive["prediction_patchtst"]
        elif method == "timemixer":
            prediction = archive["prediction_timemixer"]
        elif method == "dlinear":
            prediction = archive["prediction"]
        elif method == "nbeats_graph":
            prediction = archive["prediction_nbeats_graph"]
        else:
            raise ValueError(method)
        truth = archive["truth"]
    if truth.ndim != 3 or truth.shape[1] != 24 or prediction.shape != truth.shape:
        raise ValueError(f"unexpected prediction schema in {prediction_path}: {truth.shape}, {prediction.shape}")
    expected_units = {
        "jiaxing": {"kWh per location per hour", "kWh per station-hour"},
        "shanghai": {"kW"},
        "urbanev": {"volume.csv native hourly unit", "UrbanEV published volume unit"},
    }[dataset]
    unit = result.get("unit")
    if unit is None:
        unit = (result.get("target_protocol") or {}).get("unit")
    if unit not in expected_units:
        raise ValueError(f"unit mismatch in {result_path}: {unit}")
    if method == "patchtst":
        source_exposure = result.get("protocol", {}).get("source_minibatches")
    elif method == "dlinear":
        source_exposure = result.get("source_training", {}).get("optimizer_updates")
    else:
        source_exposure = result.get("training", {}).get("source_minibatches_total")
    if source_exposure != 2048:
        raise ValueError(f"source exposure mismatch in {result_path}")
    rmse, mae = score(truth, prediction)
    recorded = result["metrics"]
    stored_rmse = recorded.get("pooled_rmse")
    stored_mae = recorded.get("pooled_mae")
    if stored_rmse is None and method == "dlinear":
        stored_rmse, stored_mae = recorded.get("pooled_rmse"), recorded.get("pooled_mae")
    if stored_rmse is None or stored_mae is None:
        raise ValueError(f"stored pooled metric missing in {result_path}")
    if not np.isclose(rmse, stored_rmse, rtol=1e-6, atol=1e-6) or not np.isclose(mae, stored_mae, rtol=1e-6, atol=1e-6):
        raise ValueError(f"stored/recomputed metric mismatch in {result_path}")
    return truth, prediction, result


def collect_rows(require_complete: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                truth, archived = existing_arrays(dataset, seed)
            except FileNotFoundError:
                archived = {}
                truth = None
            if truth is not None:
                for model, prediction in archived.items():
                    rmse, mae = score(truth, prediction)
                    rows.append({"dataset": dataset, "model": model, "seed": seed,
                                 "rmse": rmse, "mae": mae, "status": "recomputed_from_existing_predictions",
                                 "unit": {"jiaxing": "kWh proxy", "shanghai": "kW", "urbanev": "native volume"}[dataset]})
            for method in ("dlinear", "patchtst", "timemixer", "nbeats_graph"):
                try:
                    new_truth, prediction, result = new_predictions(dataset, method, seed)
                except FileNotFoundError:
                    if require_complete:
                        raise
                    continue
                if truth is not None and not np.allclose(truth, new_truth, rtol=0.0, atol=1e-6):
                    raise ValueError(f"truth arrays do not match legacy matrix for {dataset} seed {seed}")
                rmse, mae = score(new_truth, prediction)
                rows.append({"dataset": dataset, "model": method.upper(), "seed": seed,
                             "rmse": rmse, "mae": mae, "status": "recomputed_from_new_predictions",
                             "unit": {"jiaxing": "kWh proxy", "shanghai": "kW", "urbanev": "native volume"}[dataset],
                             "origins": int(new_truth.shape[0]), "horizon": int(new_truth.shape[1]),
                             "sites": int(new_truth.shape[2]),
                             "training_seconds": result.get("training", {}).get("training_seconds"),
                             "checkpoint": result.get("artifacts", {}).get("checkpoint", result.get("checkpoint_file"))})
    if require_complete:
        new_run_rows = [row for row in rows if row["model"] in {"DLINEAR", "PATCHTST", "TIMEMIXER", "NBEATS_GRAPH"}]
        if len(new_run_rows) != 36:
            raise ValueError(f"expected 36 completed new runs, found {len(new_run_rows)}")
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "model", "seed", "rmse", "mae", "unit", "status",
               "origins", "horizon", "sites", "training_seconds", "checkpoint"]
    with (OUT / "verified_metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        models = sorted({row["model"] for row in subset})
        for model in models:
            selected = [row for row in subset if row["model"] == model]
            if len(selected) != len(SEEDS):
                continue
            summary_rows.append({"dataset": dataset, "model": model,
                                 "seed_count": len(selected),
                                 "rmse_mean_over_seeds": float(np.mean([row["rmse"] for row in selected])),
                                 "rmse_sd_over_seeds": float(np.std([row["rmse"] for row in selected], ddof=1)),
                                 "mae_mean_over_seeds": float(np.mean([row["mae"] for row in selected])),
                                 "mae_sd_over_seeds": float(np.std([row["mae"] for row in selected], ddof=1)),
                                 "unit": selected[0]["unit"]})
    summary_columns = ["dataset", "model", "seed_count", "rmse_mean_over_seeds", "rmse_sd_over_seeds",
                       "mae_mean_over_seeds", "mae_sd_over_seeds", "unit"]
    with (OUT / "summary_by_dataset.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# External strong-baseline results (2026-10-28)", "",
        "Metrics were recomputed from saved truth/prediction arrays. No cross-dataset RMSE/MAE pooling is performed.",
        "Seed means and SDs summarize the three fixed training seeds within each dataset; they are not city-level population uncertainty.",
        "The Exact/SCPRC comparison rows are from the archived 2026-10-24 external matrix (SCPRC-v2.2), not the newer homogeneous CHARGED 2026-10-27 matrix.",
        "Jiaxing uses a session-derived hourly-energy proxy (kWh); Shanghai is published station power (kW); UrbanEV retains its native volume unit.",
        "All experiments are retrospective. No external test score is used for training, scaling, or model selection.", "",
    ]
    for dataset in DATASETS:
        lines.extend([f"## {dataset}", "", "| Model | RMSE mean ± SD | MAE mean ± SD | n seeds |", "|---|---:|---:|---:|"])
        for row in summary_rows:
            if row["dataset"] != dataset:
                continue
            lines.append(f"| {row['model']} | {row['rmse_mean_over_seeds']:.4f} ± {row['rmse_sd_over_seeds']:.4f} | "
                         f"{row['mae_mean_over_seeds']:.4f} ± {row['mae_sd_over_seeds']:.4f} | {row['seed_count']} |")
        lines.append("")
    lines.extend([
        "## Protocol interpretation", "",
        "- PatchTST uses the pinned local official adapter, one volume channel, source-only six-city CHARGED training, 2048 source minibatches × 4 windows, and PCGrad over source-city tasks (342 parameter updates).",
        "- TimeMixer uses the vendored official CD route, volume-only input, source-only six-city CHARGED training, 2048 minibatches × 4 windows (2048 optimizer updates).",
        "- N-BEATS graph control uses the local shared N-BEATS-style backbone plus history-conditioned behavior-graph residual, source-only six-city CHARGED training, and 2048 minibatches × 4 windows. It is not a scprc of the cited EV paper; its objective and graph structure differ from SCPRC.",
        "- DLinear uses the local shared univariate LTSF-Linear implementation, source-only six-city CHARGED training, 2048 updates × 4 windows. It is an additional external control, not a neural strong-baseline family equivalent to PatchTST/TimeMixer.",
        "- Exposure is matched as 8192 sampled source windows, but optimizer steps differ across PCGrad PatchTST and single-task controls; do not claim optimizer-update identity.",
        "- Report each dataset separately. Historical candidate/release rows are lineage context only and are not treated as the new SCPRC-Online protocol.",
        "- `paired_block_bootstrap_168origin.csv` gives within-dataset paired moving-block intervals (168 consecutive forecast origins), conditional on the three fixed trained seeds; this is not population uncertainty over cities.",
        "- The graph control is a local N-BEATS-style behavior-graph baseline. It does not establish superiority over adaptive Graph WaveNet/MTGNN, and it is not a faithful scprc of a named N-BEATS EV paper.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = collect_rows(args.require_complete)
    write_report(rows)
    print(f"wrote {len(rows)} seed-level rows to {OUT}")


if __name__ == "__main__":
    main()

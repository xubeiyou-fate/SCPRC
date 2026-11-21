#!/usr/bin/env python3
"""Energy-operation scenario replay on the frozen CHARGED 18-fold archive.

No model is retrained.  The script consumes the canonical SCPRC prediction
caches, reconstructs no-training seasonal rules from the prefix-only CHARGED
archive, and evaluates capacity reservation/risk under fixed, pre-specified
safety margins.  All outputs stay in this experiment directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
OUT = ROOT / "experiments/supplementary/energy_scenario_20261030"
PRED_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"
ARCHIVE = ROOT / "scprc/data_adapters/charged/Hourly.zip"

CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
STARTS = np.arange(2928, 3649, dtype=np.int64)
N_ORIGINS = 721
HORIZON = 24
CONTEXT = 168
MARGINS = (0.00, 0.05, 0.10, 0.20)
PRIMARY_MARGIN = 0.10
PENALTIES = (1.0, 5.0, 10.0, 20.0)
RNG_SEED = 20261030

MODEL_KEYS = {
    "Exact PhaseFormer": "pred_exact_pcgrad_raw",
    "Full SCPRC candidate": "pred_scpcr_candidate_raw",
    "Seasonal naive 24 h": "seasonal24_naive",
    "Seasonal naive 168 h": "seasonal168_naive",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_city(city: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read only April-August prefix; September outcome rows are not read."""
    with zipfile.ZipFile(ARCHIVE) as z:
        with z.open(f"{city}/volume.csv") as f:
            frame = pd.read_csv(f, nrows=3672)
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="raise"))
    expected = pd.date_range("2023-04-01 00:00:00", periods=3672, freq="h")
    if not times.equals(expected):
        raise ValueError(f"{city}: unexpected hourly grid")
    vals = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if vals.shape[0] != 3672 or not np.isfinite(vals).all():
        raise ValueError(f"{city}: invalid volume values")
    return vals, tuple(str(x) for x in frame.columns[1:])


def load_fold(city: str, seed: int, city_cache: dict[str, tuple[np.ndarray, tuple[str, ...]]]) -> dict:
    p = PRED_ROOT / f"target_{city.lower()}_seed{seed}" / "predictions.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        starts = z["label_starts"].astype(np.int64)
        truth = z["truth_raw"].astype(np.float32)
        if not np.array_equal(starts, STARTS):
            raise ValueError(f"{p}: label_starts mismatch")
        if truth.shape[:2] != (N_ORIGINS, HORIZON):
            raise ValueError(f"{p}: truth shape mismatch {truth.shape}")
        exact = z[MODEL_KEYS["Exact PhaseFormer"]].astype(np.float32)
        full = z[MODEL_KEYS["Full SCPRC candidate"]].astype(np.float32)
        if exact.shape != truth.shape or full.shape != truth.shape:
            raise ValueError(f"{p}: model/truth shape mismatch")
    vals, sites = city_cache[city]
    if truth.shape[2] != vals.shape[1]:
        raise ValueError(f"{city} site mismatch: {truth.shape[2]} vs {vals.shape[1]}")
    # Revalidate archived truth for each seed against the prefix-only CHARGED archive.
    archive_truth = np.stack([vals[s:s + HORIZON] for s in STARTS], axis=0)
    if not np.allclose(archive_truth, truth, rtol=2e-5, atol=2e-5):
        raise ValueError(f"{city}/{seed}: prediction truth does not match CHARGED archive")
    seasonal24 = np.stack([vals[s - 24:s, :] for s in STARTS], axis=0).astype(np.float32)
    seasonal168 = np.stack([vals[s - 168:s - 144, :] for s in STARTS], axis=0).astype(np.float32)
    # Both no-training rules are causal and use only the prior 168 observed rows.
    return {
        "city": city,
        "seed": seed,
        "prediction_path": str(p),
        "prediction_sha256": sha256(p),
        "sites": sites,
        "truth": truth,
        "Exact PhaseFormer": np.maximum(exact, 0.0),
        "Full SCPRC candidate": np.maximum(full, 0.0),
        "Seasonal naive 24 h": np.maximum(seasonal24, 0.0),
        "Seasonal naive 168 h": np.maximum(seasonal168, 0.0),
    }


def capacity_metrics(truth_agg: np.ndarray, pred_agg: np.ndarray, margin: float, penalty: float) -> dict[str, float]:
    """Compute fixed-capacity metrics per origin; capacity is max 24 h forecast."""
    capacity = np.max(pred_agg, axis=1) * (1.0 + margin)
    actual_peak = np.max(truth_agg, axis=1)
    overload = np.maximum(truth_agg - capacity[:, None], 0.0)
    idle = np.maximum(capacity - actual_peak, 0.0)
    actual_energy = np.sum(truth_agg, axis=1)
    cost_raw = capacity + penalty * np.sum(overload, axis=1)
    denom = np.maximum(actual_peak, 1e-12)
    cost_norm = cost_raw / denom
    return {
        "origins": int(len(capacity)),
        "reserved_capacity_mean": float(np.mean(capacity)),
        "reserved_capacity_p95": float(np.quantile(capacity, 0.95)),
        "actual_peak_mean": float(np.mean(actual_peak)),
        "actual_peak_p95": float(np.quantile(actual_peak, 0.95)),
        "capacity_to_actual_peak_ratio_pct": float(np.mean(capacity / denom) * 100.0),
        "overload_window_rate_pct": float(np.mean(np.any(overload > 0.0, axis=1)) * 100.0),
        "overload_hours_mean": float(np.mean(np.sum(overload > 0.0, axis=1))),
        "overload_hours_total": float(np.sum(overload > 0.0)),
        "overload_volume_mean": float(np.mean(np.sum(overload, axis=1))),
        "overload_volume_total": float(np.sum(overload)),
        "idle_capacity_mean": float(np.mean(idle)),
        "idle_capacity_p95": float(np.quantile(idle, 0.95)),
        "overreserve_ratio_pct": float(np.mean(idle / denom) * 100.0),
        "actual_energy_mean": float(np.mean(actual_energy)),
        "relative_cost_raw_mean": float(np.mean(cost_raw)),
        "relative_cost_normalized_mean": float(np.mean(cost_norm)),
        "relative_cost_normalized_p95": float(np.quantile(cost_norm, 0.95)),
    }


def horizon_errors(folds: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    models = ("Exact PhaseFormer", "Full SCPRC candidate", "Seasonal naive 24 h", "Seasonal naive 168 h")
    # Compute cumulative horizon sums once per fold/model instead of materializing
    # 24 separate 42-million-element error arrays.
    accum: dict[str, dict[str, list[float]]] = {m: {"rmse": [], "mae": []} for m in models}
    for model in models:
        for fold in folds:
            err = fold[model] - fold["truth"]
            sq_by_hour = np.sum(err * err, axis=(0, 2), dtype=np.float64)
            abs_by_hour = np.sum(np.abs(err), axis=(0, 2), dtype=np.float64)
            denom = float(err.shape[0] * err.shape[2]) * np.arange(1, HORIZON + 1, dtype=np.float64)
            accum[model]["rmse"].append(np.sqrt(np.cumsum(sq_by_hour) / denom))
            accum[model]["mae"].append(np.cumsum(abs_by_hour) / denom)
            del err, sq_by_hour, abs_by_hour
    for model in models:
        rmse = np.mean(np.stack(accum[model]["rmse"], axis=0), axis=0)
        mae = np.mean(np.stack(accum[model]["mae"], axis=0), axis=0)
        for h in range(HORIZON):
            rows.append({"model": model, "horizon_hours": h + 1,
                         "equal_fold_rmse": float(rmse[h]),
                         "equal_fold_mae": float(mae[h]), "fold_count": len(folds)})
    return pd.DataFrame(rows)


def peak_decomposition(folds: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    models = ("Exact PhaseFormer", "Full SCPRC candidate", "Seasonal naive 24 h", "Seasonal naive 168 h")
    for model in models:
        for fold in folds:
            truth_agg = fold["truth"].sum(axis=2)
            peak_threshold = np.quantile(truth_agg, 0.75, axis=1, keepdims=True)
            peak_mask = truth_agg >= peak_threshold
            argmax = np.argmax(truth_agg, axis=1)
            abs_err = np.abs(fold[model] - fold["truth"])
            peak_sum = off_sum = argmax_sum = 0.0
            peak_count = off_count = argmax_count = 0
            # Avoid broadcasting a (origins,hours,1) boolean mask into a huge
            # temporary array; index one hour at a time.
            for h in range(HORIZON):
                pm = peak_mask[:, h]
                vals = abs_err[:, h, :]
                peak_sum += float(vals[pm].sum())
                peak_count += int(pm.sum()) * vals.shape[1]
                off_sum += float(vals[~pm].sum())
                off_count += int((~pm).sum()) * vals.shape[1]
            for i, h in enumerate(argmax):
                argmax_sum += float(abs_err[i, h, :].sum())
                argmax_count += abs_err.shape[2]
            peak_mae = peak_sum / max(peak_count, 1)
            off_mae = off_sum / max(off_count, 1)
            argmax_mae = argmax_sum / max(argmax_count, 1)
            rows.append({"city": fold["city"], "seed": fold["seed"], "model": model,
                         "peak_definition": "actual aggregate city load >= within-origin 75th percentile",
                         "peak_hour_fraction_pct": float(np.mean(peak_mask) * 100.0),
                         "peak_hour_mae": float(peak_mae), "offpeak_hour_mae": float(off_mae),
                         "peak_minus_offpeak_mae": float(peak_mae - off_mae),
                         "actual_argmax_hour_mae": float(argmax_mae)})
            del abs_err
    return pd.DataFrame(rows)


def plot_outputs(capacity_df: pd.DataFrame, horizon_df: pd.DataFrame, peak_df: pd.DataFrame) -> None:
    # Capacity-risk tradeoff: equal-fold averages by model and safety margin.
    summary = capacity_df.groupby(["model", "safety_margin_pct"], as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    for model, g in summary.groupby("model"):
        ax.plot(g["reserved_capacity_mean"], g["overload_window_rate_pct"], marker="o", label=model)
        for _, row in g.iterrows():
            ax.annotate(f"{row['safety_margin_pct']:.0f}%", (row["reserved_capacity_mean"], row["overload_window_rate_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean reserved capacity (volume units)")
    ax.set_ylabel("Overload-window rate (%)")
    ax.set_title("Capacity–risk trade-off (equal-fold mean)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "capacity_risk_tradeoff.png")
    plt.close(fig)

    # Relative cost sensitivity: primary penalty values, fixed margin grid.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180)
    for model, g in capacity_df.groupby("model"):
        g2 = g[g["overload_penalty"].isin(PENALTIES)].groupby(["safety_margin_pct", "overload_penalty"], as_index=False).mean(numeric_only=True)
        for penalty, gp in g2.groupby("overload_penalty"):
            axes[0].plot(gp["safety_margin_pct"], gp["relative_cost_normalized_mean"], marker="o", label=f"{model}; λ={penalty:g}")
    axes[0].set_xlabel("Safety margin (%)")
    axes[0].set_ylabel("Normalized relative cost")
    axes[0].set_title("Cost sensitivity by safety margin")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    for model, g in capacity_df[capacity_df["safety_margin_pct"] == PRIMARY_MARGIN * 100].groupby("model"):
        gg = g.sort_values("overload_penalty")
        axes[1].plot(gg["overload_penalty"], gg["relative_cost_normalized_mean"], marker="o", label=model)
    axes[1].set_xlabel("Overload penalty λ")
    axes[1].set_ylabel("Normalized relative cost")
    axes[1].set_title(f"Penalty sensitivity at {PRIMARY_MARGIN:.0%} margin")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "relative_cost_sensitivity.png")
    plt.close(fig)

    # Horizon error curve.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=180)
    for model, g in horizon_df.groupby("model"):
        axes[0].plot(g["horizon_hours"], g["equal_fold_rmse"], marker="o", ms=3, label=model)
        axes[1].plot(g["horizon_hours"], g["equal_fold_mae"], marker="o", ms=3, label=model)
    axes[0].set_title("Horizon-wise RMSE")
    axes[1].set_title("Horizon-wise MAE")
    for ax, y in zip(axes, ["RMSE", "MAE"]):
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(y + " (raw volume)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "horizon_error_curve.png")
    plt.close(fig)

    # Peak vs off-peak error decomposition.
    peak_summary = peak_df.groupby("model", as_index=False)[["peak_hour_mae", "offpeak_hour_mae", "actual_argmax_hour_mae"]].mean()
    long = peak_summary.melt(id_vars="model", var_name="period", value_name="mae")
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=180)
    x = np.arange(len(peak_summary))
    width = 0.25
    for j, period in enumerate(["peak_hour_mae", "offpeak_hour_mae", "actual_argmax_hour_mae"]):
        ax.bar(x + (j - 1) * width, peak_summary[period], width=width, label=period.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(peak_summary["model"], rotation=20, ha="right")
    ax.set_ylabel("MAE (raw volume)")
    ax.set_title("Peak-period error decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "peak_error_decomposition.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    city_cache = {city: read_city(city) for city in CITIES}
    folds = [load_fold(city, seed, city_cache) for city in CITIES for seed in SEEDS]

    # Save source protocol before derived calculations.
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "E1 energy-operation scenario replay plus horizon/peak diagnostics",
        "no_retraining": True,
        "dataset": "CHARGED v3",
        "cities": list(CITIES),
        "seeds": list(SEEDS),
        "fold_count": len(folds),
        "origins_per_fold": N_ORIGINS,
        "horizon_hours": HORIZON,
        "context_hours": CONTEXT,
        "evaluation_window": "2023-08-01 through 2023-08-31; label starts [2928, 3649)",
        "raw_archive_rows_read_per_city": 3672,
        "raw_archive_cutoff": "2023-09-01 00:00 exclusive; no September outcome rows read",
        "model_sources": {k: v for k, v in MODEL_KEYS.items() if not v.startswith("seasonal")},
        "prediction_root": str(PRED_ROOT),
        "superseded_legacy_prediction_root": str(ROOT / "experiments/supplementary/current_scpcr_charged_20261024"),
        "prediction_cache_sha256": {f"{f['city']}_seed{f['seed']}": f["prediction_sha256"] for f in folds},
        "source_archive_sha256": sha256(ARCHIVE),
        "safety_margins": [float(x) for x in MARGINS],
        "primary_safety_margin": PRIMARY_MARGIN,
        "safety_margin_policy": "Fixed a-priori margins; no test-set tuning. Primary margin is 10%.",
        "safety_margin_source": "The 10% primary margin is a pre-specified operational stress parameter used for comparability, not estimated from August labels, not selected by minimizing test cost, and not back-calibrated from model errors. The 0/5/10/20% grid is a sensitivity analysis.",
        "test_tuning_prohibited": True,
        "overload_penalties": [float(x) for x in PENALTIES],
        "relative_cost_definition": "cost = reserved_capacity + lambda * overload_volume; normalized cost divides per-origin cost by actual aggregate peak",
        "capacity_definition": "max over 24 forecasted aggregate city-volume values times (1+safety_margin)",
        "oracle_boundary": "Oracle actual peak uses realized aggregate 24-hour peak as capacity; it is an unavailable lower-bound comparator, never a deployable forecast and never used to choose the safety margin.",
        "overload_definition": "positive(actual aggregate volume - reserved capacity) per hour",
        "peak_definition": "actual aggregate city load at or above within-origin 75th percentile",
        "primary_models": ["Exact PhaseFormer", "Full SCPRC candidate", "Seasonal naive 24 h"],
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "command.txt").write_text("python experiments/supplementary/energy_scenario/run_energy_scenario.py\n", encoding="utf-8")

    # Per-fold and per-origin capacity scenario tables.
    metric_rows: list[dict] = []
    origin_rows: list[dict] = []
    for fold in folds:
        truth_agg = fold["truth"].sum(axis=2)
        for model in MODEL_KEYS:
            pred_agg = fold[model].sum(axis=2) if model != "Seasonal naive 24 h" and model != "Seasonal naive 168 h" else fold[model].sum(axis=2)
            for margin in MARGINS:
                for penalty in PENALTIES:
                    m = capacity_metrics(truth_agg, pred_agg, margin, penalty)
                    row = {"city": fold["city"], "seed": fold["seed"], "model": model,
                           "safety_margin_pct": margin * 100.0, "overload_penalty": penalty, **m}
                    metric_rows.append(row)
                    capacity = np.max(pred_agg, axis=1) * (1.0 + margin)
                    actual_peak = np.max(truth_agg, axis=1)
                    overload = np.maximum(truth_agg - capacity[:, None], 0.0)
                    idle = np.maximum(capacity - actual_peak, 0.0)
                    raw_cost = capacity + penalty * np.sum(overload, axis=1)
                    for i in range(N_ORIGINS):
                        origin_rows.append({"city": fold["city"], "seed": fold["seed"], "origin_index": int(STARTS[i]),
                                            "origin_offset": i, "model": model, "safety_margin_pct": margin * 100.0,
                                            "overload_penalty": penalty, "reserved_capacity": float(capacity[i]),
                                            "actual_peak": float(actual_peak[i]), "idle_capacity": float(idle[i]),
                                            "overload_hours": int(np.sum(overload[i] > 0.0)),
                                            "overload_volume": float(np.sum(overload[i])),
                                            "overload_window": bool(np.any(overload[i] > 0.0)),
                                            "relative_cost_raw": float(raw_cost[i]),
                                            "relative_cost_normalized": float(raw_cost[i] / max(actual_peak[i], 1e-12))})
    capacity_df = pd.DataFrame(metric_rows)
    capacity_df.to_csv(OUT / "capacity_scenario_per_fold.csv", index=False, float_format="%.8f")
    pd.DataFrame(origin_rows).to_csv(OUT / "capacity_scenario_per_origin.csv", index=False, float_format="%.8f")

    # Equal-fold summaries by model, margin, and overload penalty.
    summary = capacity_df.groupby(["model", "safety_margin_pct", "overload_penalty"], as_index=False).mean(numeric_only=True)
    summary = summary.drop(columns=["seed"], errors="ignore")
    summary["fold_count"] = len(folds)
    summary.to_csv(OUT / "capacity_scenario_equal_fold_summary.csv", index=False, float_format="%.8f")
    primary = summary[(summary["safety_margin_pct"] == PRIMARY_MARGIN * 100.0) & (summary["overload_penalty"] == 10.0)].copy()
    # Oracle is an unavailable lower-bound comparator: it sets capacity to the
    # realized aggregate 24-hour peak, so overload and idle capacity are zero.
    # It never uses predictions and is never considered deployable.
    oracle = {
        "model": "Oracle actual peak (unavailable)",
        "safety_margin_pct": 0.0, "overload_penalty": 10.0, "origins": float(N_ORIGINS),
        "reserved_capacity_mean": float(capacity_df["actual_peak_mean"].mean()),
        "reserved_capacity_p95": float(capacity_df["actual_peak_p95"].mean()),
        "actual_peak_mean": float(capacity_df["actual_peak_mean"].mean()),
        "actual_peak_p95": float(capacity_df["actual_peak_p95"].mean()),
        "capacity_to_actual_peak_ratio_pct": 100.0, "overload_window_rate_pct": 0.0,
        "overload_hours_mean": 0.0, "overload_hours_total": 0.0,
        "overload_volume_mean": 0.0, "overload_volume_total": 0.0,
        "idle_capacity_mean": 0.0, "idle_capacity_p95": 0.0, "overreserve_ratio_pct": 0.0,
        "actual_energy_mean": float(capacity_df["actual_energy_mean"].mean()),
        "relative_cost_raw_mean": float(capacity_df["actual_peak_mean"].mean()),
        "relative_cost_normalized_mean": 1.0, "relative_cost_normalized_p95": 1.0,
        "fold_count": len(folds),
    }
    primary = pd.concat([primary, pd.DataFrame([oracle])], ignore_index=True)
    primary.to_csv(OUT / "capacity_scenario_primary.csv", index=False, float_format="%.8f")

    # Error diagnostics.
    hdf = horizon_errors(folds)
    hdf.to_csv(OUT / "horizon_error_curve.csv", index=False, float_format="%.8f")
    pdf = peak_decomposition(folds)
    pdf.to_csv(OUT / "peak_error_decomposition.csv", index=False, float_format="%.8f")
    pdf.groupby("model", as_index=False).mean(numeric_only=True).to_csv(OUT / "peak_error_decomposition_equal_fold.csv", index=False, float_format="%.8f")

    plot_outputs(capacity_df, hdf, pdf)

    # Human-readable report with primary numbers and caveats.
    p = primary.copy()
    p = p.sort_values(["model"])
    lines = [
        "# E1 Energy-operation scenario replay (CHARGED 18-fold)", "",
        "Status: completed read-only replay; no model retraining. All August labels were already exposed during development, so this is retrospective evidence rather than blind confirmation.", "",
        "## Protocol", "",
        f"- Six cities × three seeds = {len(folds)} city×seed folds; 721 origins/fold; 24-hour horizon; 168-hour context.",
        "- Predictions come from the canonical homogeneous 2026-10-27 matrix; Exact and Full SCPRC use `pred_exact_pcgrad_raw` and `pred_scpcr_candidate_raw`.",
        "- Seasonal naive rules are reconstructed causally from the preceding 168 observed hours in the pinned CHARGED archive.",
        f"- Primary scenario: {PRIMARY_MARGIN:.0%} a-priori safety margin and overload penalty λ=10; margins {', '.join(f'{x:.0%}' for x in MARGINS)} and λ values {', '.join(str(x) for x in PENALTIES)} are sensitivity analyses.",
        "- City capacity is `max_24h(predicted aggregate volume) × (1 + margin)`. Overload volume is the positive hourly excess over that fixed capacity.",
        "- Relative cost is `reserved_capacity + λ × overload_volume`; normalized relative cost divides each origin's cost by its actual aggregate peak. No monetary prices or physical kW conversion are assumed.",
        "- Oracle boundary: `Oracle actual peak (unavailable)` uses the realized peak itself, hence zero overload and zero idle capacity. It is a diagnostic lower bound and cannot be used in deployment or tuning.", "",
        "## Primary capacity/risk results (equal mean over 18 folds; margin=10%, λ=10)", "",
        "| Model | Reserved capacity | Actual peak | Capacity/peak (%) | Overload-window rate (%) | Overload hours | Overload volume | Idle capacity | Normalized cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in p.iterrows():
        lines.append(f"| {r['model']} | {r['reserved_capacity_mean']:.4f} | {r['actual_peak_mean']:.4f} | {r['capacity_to_actual_peak_ratio_pct']:.2f} | {r['overload_window_rate_pct']:.2f} | {r['overload_hours_mean']:.3f} | {r['overload_volume_mean']:.4f} | {r['idle_capacity_mean']:.4f} | {r['relative_cost_normalized_mean']:.4f} |")
    lines += ["", "## Outputs", "", "- `capacity_scenario_per_origin.csv`: origin-level capacity, overload, idle-capacity, and relative-cost values.", "- `capacity_scenario_per_fold.csv`: fold-level summaries for all margin/penalty settings.", "- `capacity_scenario_equal_fold_summary.csv`: equal-fold means across the 18 city×seed folds.", "- `horizon_error_curve.csv`: 1–24 h RMSE/MAE curves.", "- `peak_error_decomposition.csv`: within-origin 75th-percentile peak-hour versus off-peak errors.", "- `capacity_risk_tradeoff.png`, `relative_cost_sensitivity.png`, `horizon_error_curve.png`, `peak_error_decomposition.png`: publication-draft diagnostics.", "", "## Interpretation boundary", "", "These metrics quantify a deployment-style capacity reservation proxy in the dataset's native volume units. Without site capacity, electricity price, or overload damage data, the relative-cost curves should not be reported as monetary savings or grid reliability guarantees. The model can reduce or increase capacity risk depending on city, seed, safety margin, and penalty; report the city-level table and sensitivity rather than a single universal dominance claim.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(OUT), "folds": len(folds), "primary_rows": int(len(primary))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

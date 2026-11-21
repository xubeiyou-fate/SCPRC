#!/usr/bin/env python3
"""Strict source-only transfer from CHARGED to the frozen Shanghai MP-EVData panel.

The release decision and support radius are calibrated from nested CHARGED
source-only July cross-fits. Shanghai Q4 labels are read only after the
per-seed source-only decision and model checkpoints have been frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
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
EXP = ROOT / 'scprc/training'
DATA = ROOT / "scprc/data_adapters/public/mp_evdata_figshare_29882366"
OUT = EXP / "mp_evdata_scpcr_external_20261021"
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scprc.data_adapters.charged.city_adapter import ARCHIVE, CITY_CODES  # noqa: E402
import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.source_calibration_support as cert  # noqa: E402
import scprc.training.scprc_training_support as v21  # noqa: E402
import scprc.training.train_scprc as v22  # noqa: E402
from scprc.baselines.patchtst_adapter import PatchTSTUrbanEV, UPSTREAM_COMMIT  # noqa: E402

SEEDS = (13, 17, 29)
STEPS = 2048
BATCH_SIZE = protocol.BATCH_SIZE
TARGET_PREFIX_END = 6576              # 2024-10-01 00:00, exclusive
TARGET_TEST_END = 8784                # 2025-01-01 00:00, exclusive
TARGET_PROFILE_START = 2184           # 2024-04-01, same four-month span as source
TARGET_PROFILE_END = 5112             # 2024-08-01, exclusive
TAIL_TOLERANCE = 0.02
TAIL_QUANTILE = 0.95
SUPPORT_QUANTILE = 0.90
MIN_SUPPORTED = 3
SITES = ("A1", "A2", "A3", "A4", "A5", "A6", "A9")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_source_train(archive_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Materialize only Apr-Jul source labels, exactly the source training span."""
    import zipfile

    cities: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for city in CITY_CODES:
            member = f"{city}/volume.csv"
            with zf.open(member) as f:
                frame = pd.read_csv(f, nrows=protocol.TRAIN_END)
            if len(frame) != protocol.TRAIN_END or frame.shape[1] < 2:
                raise ValueError(f"unexpected source prefix shape for {member}: {frame.shape}")
            timestamps = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            expected = pd.date_range(protocol.FIRST_HOUR, periods=protocol.TRAIN_END, freq="h")
            if not timestamps.equals(expected):
                raise ValueError(f"invalid source timestamps for {member}")
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
            site_ids = tuple(map(str, frame.columns[1:]))
            if not site_ids or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"invalid source values in {member}")
            cities[city] = {"city_id": city, "times": timestamps,
                            "site_ids": site_ids, "values": values}
    return cities, {
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": sha256(archive_path),
        "source_cities": list(CITY_CODES),
        "source_rows_per_city": protocol.TRAIN_END,
        "source_interval": "2023-04-01 through 2023-07-31",
        "source_august_labels_materialized": False,
    }


def source_only_calibration(seed: int, nested_root: Path) -> dict[str, Any]:
    """Pool the six pseudo-targets across their five nested source folds."""
    grouped: dict[str, list[dict[str, Any]]] = {c: [] for c in CITY_CODES}
    for outer in CITY_CODES:
        p = nested_root / f"outer_{outer.lower()}_seed{seed}" / "result.json"
        if not p.is_file():
            raise FileNotFoundError(f"nested source-only calibration is incomplete: {p}")
        result = json.loads(p.read_text())
        if result.get("status") != "completed" or result.get("seed") != seed or result.get("outer_city") != outer:
            raise ValueError(f"invalid nested calibration result: {p}")
        for row in result.get("inner_crossfit", []):
            if row.get("outer_target_in_training_or_calibration") is not False:
                raise ValueError(f"outer city leaked into inner calibration in {p}")
            grouped[row["pseudo_target"]].append({**row, "_outer_city": outer})
    pseudo = []
    for city, rows in grouped.items():
        if len(rows) != len(CITY_CODES) - 1:
            raise ValueError(f"expected five independent nested pseudo-target results for {city}, got {len(rows)}")
        pseudo.append({
            "pseudo_target": city,
            "support_ratio": float(np.mean([r["support_ratio"]["ratio"] for r in rows])),
            "baseline": {k: float(np.mean([r["baseline_july"][k] for r in rows]))
                         for k in ("rmse", "mae", "window_rmse_p95")},
            "candidate": {k: float(np.mean([r["candidate_july"][k] for r in rows]))
                          for k in ("rmse", "mae", "window_rmse_p95")},
            "nested_outer_folds": [r["_outer_city"] for r in rows],
        })
    radius = float(np.quantile([r["support_ratio"] for r in pseudo], SUPPORT_QUANTILE, method="linear"))
    supported = [r for r in pseudo if r["support_ratio"] <= radius]
    if len(supported) < MIN_SUPPORTED:
        gate_passed = False
        gate_reason = "fewer than three source pseudo-targets fall inside the fixed q90 radius"
    else:
        base_rmse = float(np.mean([r["baseline"]["rmse"] for r in supported]))
        cand_rmse = float(np.mean([r["candidate"]["rmse"] for r in supported]))
        base_mae = float(np.mean([r["baseline"]["mae"] for r in supported]))
        cand_mae = float(np.mean([r["candidate"]["mae"] for r in supported]))
        wins = sum(r["candidate"]["rmse"] <= r["baseline"]["rmse"] and
                   r["candidate"]["mae"] <= r["baseline"]["mae"] for r in supported)
        tail = all(r["candidate"]["window_rmse_p95"] <=
                   r["baseline"]["window_rmse_p95"] * (1 + TAIL_TOLERANCE) for r in supported)
        rmse_ok, mae_ok = cand_rmse <= base_rmse, cand_mae <= base_mae
        wins_ok = wins >= max(3, math.ceil(len(supported) / 2))
        gate_passed = bool(rmse_ok and mae_ok and wins_ok and tail)
        gate_reason = "all fixed source-only criteria passed" if gate_passed else "fixed source-only RMSE/MAE/win/p95 rule rejected candidate"
    return {
        "calibration_source": "six CHARGED cities; five nested four-source models per pseudo-target; July labels only",
        "pseudo_target_averaging": "equal-weight mean across the five outer folds in which this city was the inner pseudo-target",
        "support_radius_ratio": radius,
        "support_radius_quantile": SUPPORT_QUANTILE,
        "support_radius_method": "linear empirical q90 across six equal-weight pseudo-city mean profile ratios",
        "pseudo_targets": pseudo,
        "supported_pseudo_targets": [r["pseudo_target"] for r in supported],
        "supported_pseudo_target_count": len(supported),
        "baseline_macro_rmse": float(np.mean([r["baseline"]["rmse"] for r in supported])) if supported else None,
        "candidate_macro_rmse": float(np.mean([r["candidate"]["rmse"] for r in supported])) if supported else None,
        "baseline_macro_mae": float(np.mean([r["baseline"]["mae"] for r in supported])) if supported else None,
        "candidate_macro_mae": float(np.mean([r["candidate"]["mae"] for r in supported])) if supported else None,
        "both_metric_wins": sum(r["candidate"]["rmse"] <= r["baseline"]["rmse"] and r["candidate"]["mae"] <= r["baseline"]["mae"] for r in supported),
        "gate_passed": bool(gate_passed),
        "gate_reason": gate_reason,
        "fixed_gate_rule": "supported source pseudo-target macro RMSE/MAE non-worse; at least max(3, ceil(n/2)) pseudo-targets win both; each supported p95 window-RMSE <= exact x 1.02",
        "gate_parameter_search": False,
        "expert_pool": False,
    }


def profile_ratio(source_cities: dict[str, dict[str, Any]], target_window: np.ndarray) -> dict[str, float]:
    profiles = np.stack([cert._profile(source_cities[c]["values"]) for c in CITY_CODES])
    target_profile = cert._profile(target_window)
    center = profiles.mean(axis=0)
    spread = profiles.std(axis=0) + 1e-6
    dists = np.linalg.norm((profiles - center) / spread, axis=1)
    target_distance = float(np.linalg.norm((target_profile - center) / spread))
    max_source = float(dists.max())
    return {"target_profile_distance": target_distance,
            "source_profile_max_distance": max_source,
            "ratio": target_distance / max(max_source, 1e-12),
            "profile_interval": "2024-04-01 through 2024-07-31 (same duration/calendar span as CHARGED source training)"}


def _metric(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    err = pred.astype(np.float64) - truth.astype(np.float64)
    win = np.sqrt(np.mean(err ** 2, axis=(1, 2)))
    per_site = {}
    for n, site in enumerate(SITES):
        e = err[:, :, n]
        per_site[site] = {"rmse_kw": float(np.sqrt(np.mean(e ** 2))),
                          "mae_kw": float(np.mean(np.abs(e)))}
    return {"rmse_kw": float(np.sqrt(np.mean(err ** 2))),
            "mae_kw": float(np.mean(np.abs(err))),
            "macro_site_rmse_kw": float(np.mean([m["rmse_kw"] for m in per_site.values()])),
            "macro_site_mae_kw": float(np.mean([m["mae_kw"] for m in per_site.values()])),
            "window_rmse_p95_kw": float(np.quantile(win, TAIL_QUANTILE)),
            "per_site": per_site}


def _train_patchtst(cities: dict, source_ids: tuple[str, ...], mean: float, std: float,
                    *, seed: int, steps: int, device: str, torch_threads: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = PatchTSTUrbanEV(seq_len=protocol.CONTEXT, pred_len=protocol.HORIZON,
                            channels=1, patch_len=16, stride=8, d_model=32,
                            d_ff=64, e_layers=1, n_heads=4, dropout=0.0,
                            fc_dropout=0.0, head_dropout=0.0, revin=True).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    starts = protocol._train_starts()
    rng = np.random.default_rng(seed)
    updates_by_city = {c: 0 for c in source_ids}
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        city = source_ids[step % len(source_ids)]
        x, y = protocol._make_batch(cities[city]["values"], starts, mean, std, rng, BATCH_SIZE)
        batch, _, nodes = x.shape
        x = x.reshape(batch * nodes, protocol.CONTEXT, 1).to(device)
        y = y.reshape(batch * nodes, protocol.HORIZON, 1).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        if not torch.isfinite(pred).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite PatchTST output/loss at update {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates_by_city[city] += 1
    return model, {"steps": steps, "batch_size_windows": BATCH_SIZE,
                   "updates_by_city": updates_by_city, "optimizer": "AdamW",
                   "learning_rate": 1e-3, "weight_decay": 1e-4,
                   "loss": "source-standardized MSE", "revin": True,
                   "upstream_commit": UPSTREAM_COMMIT,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "training_seconds": time.perf_counter() - started}


def _predict_patchtst(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
                      mean: float, std: float, *, device: str) -> np.ndarray:
    model.eval(); chunks = []
    with torch.no_grad():
        for offset in range(0, len(starts), 128):
            current = starts[offset:offset + 128]
            raw = np.stack([values[s - protocol.CONTEXT:s] for s in current]).astype(np.float32)
            b, _, nodes = raw.shape
            x = torch.from_numpy((raw - mean) / std).reshape(b * nodes, protocol.CONTEXT, 1).to(device)
            pred = model(x).cpu().numpy().reshape(b, protocol.HORIZON, nodes)
            chunks.append(pred * std + mean)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _naive_predictions(values: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    persist, daily, weekly = [], [], []
    for s in starts:
        history = values[s - protocol.CONTEXT:s]
        persist.append(np.repeat(history[-1:, :], protocol.HORIZON, axis=0))
        # Mean of the previous seven observations at the same hour-of-day.
        recent = np.stack([values[s + k - 24 * np.arange(1, 8), :] for k in range(protocol.HORIZON)])
        daily.append(recent.mean(axis=1))
        weekly.append(values[s - 168:s - 144])
    return {"persistence": np.stack(persist).astype(np.float32),
            "daily_mean_7d": np.stack(daily).astype(np.float32),
            "weekly_naive": np.stack(weekly).astype(np.float32)}


def _open_q4_after_freeze(raw: Path, prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected = pd.date_range("2024-01-01 00:00", periods=TARGET_TEST_END, freq="h")
    q4 = []
    for site in SITES:
        # Materialize exactly through 2024-12-31. Jan 2025 rows are not opened.
        frame = pd.read_excel(raw, sheet_name=site, nrows=TARGET_TEST_END)
        times = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="coerce"))
        if not times.equals(expected):
            raise ValueError(f"post-freeze Q4 timestamp audit failed for {site}")
        vals = pd.to_numeric(frame["power"], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError(f"invalid Q4 load for {site}")
        q4.append(vals[TARGET_PREFIX_END:TARGET_TEST_END])
    truth = np.stack(q4, axis=1)
    values = np.concatenate([prefix, truth], axis=0)
    starts = np.arange(TARGET_PREFIX_END, TARGET_TEST_END - protocol.HORIZON + 1, dtype=np.int64)
    if truth.shape != (TARGET_TEST_END - TARGET_PREFIX_END, len(SITES)) or starts.size != 2185:
        raise AssertionError("Q4 panel shape/origin check failed")
    return values, starts


def run_seed(seed: int, *, steps: int = STEPS, torch_threads: int = 1,
             device: str = "cuda", archive_path: Path = ARCHIVE,
             nested_root: Path | None = None) -> dict[str, Any]:
    if seed not in SEEDS: raise ValueError(f"seed must be one of {SEEDS}")
    out = OUT / f"seed{seed}"
    if (out / "result.json").is_file():
        old = json.loads((out / "result.json").read_text())
        if old.get("status") == "completed" and old.get("seed") == seed:
            return old
    out.mkdir(parents=True, exist_ok=False)
    nested_root = nested_root or (EXP / "charged_scpcr_v22_nested_source_calibration_20261021")
    freeze_meta = json.loads((OUT / "protocol_freeze.json").read_text())
    raw = DATA / "station-level load Profile 1h.xlsx"
    if sha256(raw) != freeze_meta["raw_sha256"]:
        raise ValueError("raw external workbook changed after protocol freeze")
    panel = np.load(OUT / "mp_evdata_history_prefix.npz", allow_pickle=False)
    if tuple(panel["site_ids"].tolist()) != SITES:
        raise ValueError("prepared target panel site order differs from frozen protocol")
    target_history = panel["history_kw"].astype(np.float32)
    if target_history.shape != (TARGET_PREFIX_END, len(SITES)):
        raise ValueError("unexpected prepared target prefix shape")

    calibration = source_only_calibration(seed, nested_root)
    source_cities, source_provenance = load_source_train(archive_path)
    source_ids = tuple(CITY_CODES)
    mean, std = protocol.source_scaler(source_cities, source_ids)
    support = profile_ratio(source_cities,
                            target_history[TARGET_PROFILE_START:TARGET_PROFILE_END])
    support_pass = support["ratio"] <= calibration["support_radius_ratio"]
    use_candidate = bool(support_pass and calibration["gate_passed"])
    decision = {"support_passed": bool(support_pass),
                "inner_gate_passed": bool(calibration["gate_passed"]),
                "selected_model": "source_certified_internal" if use_candidate else "phaseformer_exact",
                "fallback_reason": None if use_candidate else (
                    "target profile beyond source-only q90 radius" if not support_pass
                    else "nested source-only dual-metric/p95 gate rejected candidate")}
    threshold = cert._estimate_threshold(source_cities, source_ids, mean, std,
                                         seed=seed, device=device, quantile=0.90)

    exact, exact_info = protocol._train_one(
        "phaseformer_exact", source_cities, source_ids, len(SITES), mean, std,
        seed=seed, steps=steps, device=device, torch_threads=torch_threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    candidate = v21._build_model(len(SITES), threshold, support_scale=1.0)
    candidate_info = v22._train_candidate_pcgrad(
        candidate, source_cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads)
    patch, patch_info = _train_patchtst(
        source_cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads)

    frozen = {
        "status": "frozen_before_q4_labels_materialized", "seed": seed,
        "source_only_calibration": calibration, "target_support": support,
        "release_decision": decision, "source_scaler_mean": mean,
        "source_scaler_std": std, "source_relation_threshold": threshold,
        "source_provenance": source_provenance,
        "target_history_sha256": freeze_meta["prepared_history_sha256"],
        "raw_external_workbook_sha256": freeze_meta["raw_sha256"],
        "models_trained_and_saved_before_test_read": True,
        "target_q4_labels_materialized_before_freeze": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training": {"phaseformer_exact": exact_info,
                     "scpcr_candidate": candidate_info,
                     "patchtst": patch_info},
        "no_gate_search": True, "no_expert_pool": True,
    }
    torch.save({k: v.detach().cpu() for k, v in exact.state_dict().items()}, out / "phaseformer_exact.pt")
    torch.save({k: v.detach().cpu() for k, v in candidate.state_dict().items()}, out / "scpcr_candidate.pt")
    torch.save({k: v.detach().cpu() for k, v in patch.state_dict().items()}, out / "patchtst.pt")
    (out / "freeze.json").write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")

    # Only after frozen models and release decision are on disk do we parse Q4 labels.
    target_values, starts = _open_q4_after_freeze(raw, target_history)
    truth = np.stack([target_values[s:s + protocol.HORIZON] for s in starts]).astype(np.float32)
    pred_exact = v21._predict(exact, target_values, starts, mean, std, device=device)
    pred_candidate = v21._predict(candidate, target_values, starts, mean, std, device=device)
    pred_patch = _predict_patchtst(patch, target_values, starts, mean, std, device=device)
    naive = _naive_predictions(target_values, starts)
    predictions = {"phaseformer_exact": pred_exact,
                   "scpcr_candidate": pred_candidate,
                   "scpcr_frozen_release": pred_candidate if use_candidate else pred_exact,
                   "patchtst": pred_patch, **naive}
    metrics = {name: _metric(pred, truth) for name, pred in predictions.items()}
    np.savez_compressed(out / "q4_predictions.npz", label_starts=starts,
                        site_ids=np.asarray(SITES, dtype=str), truth_kw=truth,
                        **{f"pred_{k}_kw": v.astype(np.float32) for k, v in predictions.items()})
    result = {
        "status": "completed", "experiment": "CHARGED-to-Shanghai MP-EVData source-only external-city evaluation",
        "seed": seed, "city": "Shanghai", "sites": list(SITES),
        "release_decision": decision, "support": support,
        "source_only_calibration": calibration,
        "metrics": metrics,
        "protocol": {"context_hours": protocol.CONTEXT, "horizon_hours": protocol.HORIZON,
                     "target_history_interval": "2024-01-01 through 2024-09-30",
                     "target_support_profile_interval": "2024-04-01 through 2024-07-31",
                     "target_test_interval": "2024-10-01 through 2024-12-31",
                     "target_scaling_or_optimization": False,
                     "test_origins": int(starts.size), "rolling_1h_update": True,
                     "target_q4_labels_read_after_freeze": True,
                     "training_steps_per_model": steps,
                     "baselines": ["PhaseFormer exact", "SCPRC candidate/frozen release", "source-only PatchTST", "persistence", "7-day same-hour mean", "weekly seasonal naive"]},
        "data": {"dataset_doi": freeze_meta["dataset_doi"],
                 "dataset_license": freeze_meta["dataset_license"],
                 "target_unit": "kW mean hourly station load; CHARGED kWh per one-hour interval is numerically comparable as mean kW",
                 "excluded_sheets": freeze_meta["excluded_sheets"],
                 "raw_sha256": freeze_meta["raw_sha256"]},
        "training": {"phaseformer_exact": exact_info,
                     "scpcr_candidate": candidate_info, "patchtst": patch_info},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (HERE, protocol.HERE, cert.HERE, v21.HERE, v22.HERE)},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    del exact, candidate, patch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps({"seed": seed, "status": "completed", "decision": decision,
                      "metrics": {k: {"rmse_kw": v["rmse_kw"], "mae_kw": v["mae_kw"]}
                                  for k, v in metrics.items()}}, indent=2, allow_nan=False))
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--nested-root", type=Path)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_seed(a.seed, steps=a.steps, torch_threads=a.torch_threads,
             device=device, archive_path=a.archive, nested_root=a.nested_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

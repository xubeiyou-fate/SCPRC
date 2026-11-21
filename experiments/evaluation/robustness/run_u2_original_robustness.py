#!/usr/bin/env python3
"""U2: Original PhaseFormer clean-gated robustness replay.

This runner deliberately keeps the 2026-10-30 P1 robustness protocol fixed:
the old clean gate, fixed perturbation arrays, and six scenario definitions are
read as references.  Only the model under evaluation changes to the canonical
no-PCGrad Original PhaseFormer checkpoint from p2_no_pcgrad_phaseformer.

The old P1 directory is never written.  U2 writes one independent directory
per city/seed fold and a combined Original-vs-Full summary at its root.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents
            if (parent / 'scprc/training').is_dir())
P1_ROOT = ROOT / "experiments/forecasting/p1_canonical_robustness"
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
DEFAULT_OUTPUT = ROOT / "experiments/evaluation/u2_original_robustness"
ARCHIVE = ROOT / "scprc/data_adapters/charged/Hourly.zip"
PROTOCOL_FILE = ROOT / "scprc/training/scprc_matched_protocol.json"
SCRIPT_OLD = ROOT / "experiments/input_robustness/run_p1_clean_gated_robustness.py"
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
CONTEXT = 168
HORIZON = 24
TRAIN_END = 2928
TORCH_THREADS = 2
DEVICE = "cuda"
MODEL_NAME = "original_phaseformer_no_pcgrad"
SCENARIOS = (
    "point_missing_10pct_mean_impute",
    "temporal_block_missing_24h_mean_impute",
    "site_outage_10pct_mean_impute",
    "scale_shift_plus20pct",
    "gaussian_noise_sigma10pct_source_std",
    "positive_spike_1pct_x3",
)
RESULT_FIELDS = (
    "target_city", "seed", "model", "scenario", "rmse", "mae",
    "p95_abs_error", "clean_rmse", "clean_mae", "rmse_delta", "mae_delta",
    "rmse_relative_delta_pct", "mae_relative_delta_pct",
)

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(array.view(np.uint8))
        return digest.hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RESULT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
    }


def required_files(root: Path) -> tuple[str, ...]:
    return (
        "clean_gate.json", "fixed_masks_and_noise.npz", "scenario_protocol.json",
        "robustness_results.csv", "run_manifest.json",
    )


def complete(root: Path) -> bool:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("status") == "passed"
        and manifest.get("perturbation_scores_generated") is True
        and all((root / name).is_file() for name in required_files(root))
    )


def load_original_fold(target: str, seed: int) -> dict[str, Any]:
    fold = ORIGINAL_ROOT / f"target_{target.lower()}_seed{seed}"
    result_path = fold / "result.json"
    prediction_path = fold / "predictions.npz"
    checkpoint_path = fold / "phaseformer_no_pcgrad.pt"
    if not all(path.is_file() for path in (result_path, prediction_path, checkpoint_path)):
        raise FileNotFoundError(f"missing Original fold artifacts under {fold}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {"label_starts", "truth_raw", "site_ids", "pred_phaseformer_no_pcgrad_raw"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"Original predictions missing {sorted(missing)}")
        prediction = {key: archive[key].copy() for key in required}
    return {
        "fold": fold,
        "result_path": result_path,
        "prediction_path": prediction_path,
        "checkpoint_path": checkpoint_path,
        "result": result,
        "prediction": prediction,
    }


def load_old_reference(target: str, seed: int) -> dict[str, Any]:
    root = P1_ROOT / f"target_{target.lower()}_seed{seed}"
    required = required_files(root)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing old P1 reference files under {root}: {missing}")
    gate = json.loads((root / "clean_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    protocol_json = json.loads((root / "scenario_protocol.json").read_text(encoding="utf-8"))
    with np.load(root / "fixed_masks_and_noise.npz", allow_pickle=False) as archive:
        masks = {key: archive[key].copy() for key in archive.files}
    rows = read_csv(root / "robustness_results.csv")
    full_rows = [row for row in rows if row.get("model") == "scpcr"]
    if gate.get("status") != "passed" or not gate.get("checks", {}).get("gate_pass"):
        raise RuntimeError(f"old P1 clean gate is not passed for {target}/seed{seed}")
    if manifest.get("status") != "passed" or not manifest.get("perturbation_scores_generated"):
        raise RuntimeError(f"old P1 robustness is incomplete for {target}/seed{seed}")
    if len(full_rows) != len(SCENARIOS):
        raise RuntimeError(f"old P1 Full rows expected {len(SCENARIOS)}, got {len(full_rows)}")
    return {"root": root, "gate": gate, "manifest": manifest,
            "protocol": protocol_json, "masks": masks, "full_rows": full_rows}


def predict_model(model: torch.nn.Module, histories: np.ndarray, mean: float,
                  std: float, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(histories), 4):
            batch = (histories[offset:offset + 4] - mean) / std
            output = model(torch.from_numpy(batch).to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite forecast")
            outputs.append(output.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32) * std + mean


def build_model(n_sites: int, checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = protocol._build_model("phaseformer_exact", n_sites).float()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def make_scenarios_from_fixed_masks(histories: np.ndarray, mean: float,
                                    fixed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "point_missing_mask", "temporal_block_mask", "site_outage_mask",
        "gaussian_noise", "spike_mask",
    }
    missing = required.difference(fixed)
    if missing:
        raise KeyError(f"old fixed mask archive missing {sorted(missing)}")
    scenarios = {}
    point = histories.copy()
    point[fixed["point_missing_mask"]] = mean
    scenarios["point_missing_10pct_mean_impute"] = point
    block = histories.copy()
    block[fixed["temporal_block_mask"]] = mean
    scenarios["temporal_block_missing_24h_mean_impute"] = block
    outage = histories.copy()
    outage[fixed["site_outage_mask"]] = mean
    scenarios["site_outage_10pct_mean_impute"] = outage
    scenarios["scale_shift_plus20pct"] = np.maximum(histories * 1.20, 0.0).astype(np.float32)
    scenarios["gaussian_noise_sigma10pct_source_std"] = np.maximum(
        histories + fixed["gaussian_noise"], 0.0
    ).astype(np.float32)
    spike = histories.copy()
    spike[fixed["spike_mask"]] = spike[fixed["spike_mask"]] * 3.0
    scenarios["positive_spike_1pct_x3"] = spike
    return scenarios


def verify_fixed_protocol(old: dict[str, Any], fixed: dict[str, np.ndarray]) -> dict[str, Any]:
    descriptions = old["protocol"]
    checks = {
        "mask_seed_is_20261030": old["manifest"].get("mask_seed") == 20261030,
        "scenario_names_match": sorted(descriptions) == sorted(SCENARIOS),
        "point_mask_hash_match": descriptions["point_missing_10pct_mean_impute"].get("mask_sha256")
        == json_digest(fixed["point_missing_mask"].astype(np.uint8)),
        "temporal_block_starts_hash_match": descriptions["temporal_block_missing_24h_mean_impute"].get(
            "block_starts_sha256"
        ) == json_digest(fixed["temporal_block_starts"]),
        "site_indices_match": descriptions["site_outage_10pct_mean_impute"].get("site_indices")
        == fixed["site_outage_indices"].tolist(),
        "noise_hash_match": descriptions["gaussian_noise_sigma10pct_source_std"].get("noise_sha256")
        == json_digest(fixed["gaussian_noise"]),
        "spike_mask_hash_match": descriptions["positive_spike_1pct_x3"].get("mask_sha256")
        == json_digest(fixed["spike_mask"].astype(np.uint8)),
    }
    checks["all_fixed_protocol_checks"] = all(checks.values())
    if not checks["all_fixed_protocol_checks"]:
        raise RuntimeError(f"fixed-mask protocol verification failed: {checks}")
    return checks


def run_fold(target: str, seed: int, *, output_root: Path, device: str,
             torch_threads: int, resume: bool = True) -> dict[str, Any]:
    target = target.upper()
    fold_root = output_root / f"target_{target.lower()}_seed{seed}"
    if resume and complete(fold_root):
        manifest = json.loads((fold_root / "run_manifest.json").read_text(encoding="utf-8"))
        return {"target_city": target, "seed": seed, "status": "passed",
                "action": "skipped_existing_complete", "output_root": str(fold_root),
                "completed_scenarios": manifest.get("completed_scenarios", [])}
    fold_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "U2 Original PhaseFormer six-city clean-gated robustness",
        "status": "running", "stage": "loading_references",
        "target_city": target, "seed": seed, "model": MODEL_NAME,
        "device": device, "torch_threads": torch_threads,
        "old_p1_reference_root": str(P1_ROOT / f"target_{target.lower()}_seed{seed}"),
        "original_reference_root": str(ORIGINAL_ROOT / f"target_{target.lower()}_seed{seed}"),
        "runner": str(HERE), "runner_sha256": sha256(HERE),
        "old_runner": str(SCRIPT_OLD), "old_runner_sha256": sha256(SCRIPT_OLD),
        "protocol_file": str(PROTOCOL_FILE), "protocol_sha256": sha256(PROTOCOL_FILE),
        "archive": str(ARCHIVE), "archive_sha256": sha256(ARCHIVE),
        "context_hours": CONTEXT, "horizon_hours": HORIZON, "train_end_exclusive": TRAIN_END,
        "scenario_names": list(SCENARIOS), "scenario_count": len(SCENARIOS),
        "required_outputs": list(required_files(fold_root)),
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "perturbation_scores_generated": False, "completed_scenarios": [],
    }
    atomic_json(fold_root / "run_manifest.json", manifest)
    torch.set_num_threads(max(1, int(torch_threads)))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    original = load_original_fold(target, seed)
    old = load_old_reference(target, seed)
    fixed = old["masks"]
    fixed_checks = verify_fixed_protocol(old, fixed)
    cities, provenance = protocol.load_cities(ARCHIVE)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, values = protocol._validation_data(cities[target])
    histories = np.stack([values[start - CONTEXT:start] for start in starts]).astype(np.float32)
    saved = original["prediction"]
    metadata_checks = {
        "starts_equal": bool(np.array_equal(saved["label_starts"], starts)),
        "truth_equal": bool(np.array_equal(saved["truth_raw"], truth)),
        "site_order_equal": bool(np.array_equal(saved["site_ids"].astype(str), np.asarray(cities[target]["site_ids"], dtype=str))),
        "shape_equal": bool(saved["pred_phaseformer_no_pcgrad_raw"].shape == truth.shape),
        "prediction_finite": bool(np.isfinite(saved["pred_phaseformer_no_pcgrad_raw"]).all()),
        "archive_md5_matches_published": hashlib.md5(ARCHIVE.read_bytes()).hexdigest() == protocol.ARCHIVE_MD5,
        "timestamp_first": str(cities[target]["times"][0]),
        "timestamp_last": str(cities[target]["times"][-1]),
        "evaluation_origins": int(len(starts)),
        "site_count": int(len(saved["site_ids"])),
    }
    metadata_checks["all_metadata_checks"] = all(
        value for key, value in metadata_checks.items() if isinstance(value, bool)
    )
    model, checkpoint = build_model(len(saved["site_ids"]), original["checkpoint_path"], device)
    checkpoint_prediction = predict_model(model, histories, mean, std, device)
    saved_prediction = saved["pred_phaseformer_no_pcgrad_raw"]
    checkpoint_diff = checkpoint_prediction.astype(np.float64) - saved_prediction.astype(np.float64)
    model_checks = {
        "checkpoint_sha256": sha256(original["checkpoint_path"]),
        "checkpoint_model_name": checkpoint.get("model_name"),
        # The archived prediction was generated on CUDA; CPU replay can differ
        # by one FP32 ulp after the large-site forward pass.  Keep this
        # tolerance explicit and retain the checkpoint/data/hash checks below.
        "checkpoint_prediction_match": bool(np.max(np.abs(checkpoint_diff)) <= 2e-3),
        "checkpoint_prediction_max_abs_diff": float(np.max(np.abs(checkpoint_diff))),
        "checkpoint_prediction_mean_abs_diff": float(np.mean(np.abs(checkpoint_diff))),
        "clean_metrics": metric(saved_prediction, truth),
        "checkpoint_metrics": metric(checkpoint_prediction, truth),
    }
    model_checks["checkpoint_metrics_match"] = bool(
        abs(model_checks["clean_metrics"]["rmse"] - model_checks["checkpoint_metrics"]["rmse"]) <= 1e-4
        and abs(model_checks["clean_metrics"]["mae"] - model_checks["checkpoint_metrics"]["mae"]) <= 1e-4
    )
    gate_checks = {**metadata_checks, **fixed_checks, **{
        "old_full_clean_gate_passed": old["gate"].get("checks", {}).get("gate_pass", False),
        "original_checkpoint_prediction_match": model_checks["checkpoint_prediction_match"],
        "original_checkpoint_metrics_match": model_checks["checkpoint_metrics_match"],
    }}
    gate_checks["gate_pass"] = bool(all(
        value for key, value in gate_checks.items()
        if key not in {"timestamp_first", "timestamp_last"} and isinstance(value, bool)
    ))
    gate = {
        "status": "passed" if gate_checks["gate_pass"] else "blocked",
        "experiment": "U2 Original PhaseFormer clean gate",
        "checks": gate_checks,
        "original": model_checks,
        "reference_old_p1_clean_gate": str(old["root"] / "clean_gate.json"),
        "reference_old_p1_clean_gate_sha256": sha256(old["root"] / "clean_gate.json"),
        "reference_fixed_masks": str(old["root"] / "fixed_masks_and_noise.npz"),
        "reference_fixed_masks_sha256": sha256(old["root"] / "fixed_masks_and_noise.npz"),
        "source_scaler": {"source_cities": list(source_ids), "mean": float(mean), "std": float(std),
                          "training_end_exclusive": TRAIN_END},
        "truth_shape": list(truth.shape),
    }
    atomic_json(fold_root / "clean_gate.json", gate)
    manifest.update({"stage": "clean_gate_passed" if gate["status"] == "passed" else "clean_gate_failed",
                     "status": gate["status"], "gate": gate,
                     "original_checkpoint": str(original["checkpoint_path"]),
                     "original_predictions": str(original["prediction_path"]),
                     "source_cities": list(source_ids), "validation_origins": int(len(starts)),
                     "updated_at_utc": utc_now()})
    atomic_json(fold_root / "run_manifest.json", manifest)
    if not gate_checks["gate_pass"]:
        return {"target_city": target, "seed": seed, "status": "blocked",
                "stage": "clean_gate_failed", "output_root": str(fold_root)}

    old_mask_path = old["root"] / "fixed_masks_and_noise.npz"
    target_mask_path = fold_root / "fixed_masks_and_noise.npz"
    if not target_mask_path.is_file():
        # The fixed archive is ~1.1 GB per fold.  A hard link preserves an
        # independent output path and exact bytes without duplicating storage;
        # the old P1 file is never opened for writing by this runner.
        os.link(old_mask_path, target_mask_path)
    target_protocol_path = fold_root / "scenario_protocol.json"
    if not target_protocol_path.is_file():
        shutil.copy2(old["root"] / "scenario_protocol.json", target_protocol_path)
    scenarios = make_scenarios_from_fixed_masks(histories, mean, fixed)
    partial_path = fold_root / "robustness_results.partial.csv"
    rows = read_csv(partial_path) if resume else []
    rows = [row for row in rows if row.get("model") == MODEL_NAME and row.get("scenario") in SCENARIOS]
    completed = {row.get("scenario") for row in rows if row.get("scenario") in SCENARIOS}
    manifest.update({"stage": "scores", "status": "running", "fixed_mask_file": str(target_mask_path),
                     "scenario_protocol_file": str(target_protocol_path),
                     "completed_scenarios": sorted(completed), "updated_at_utc": utc_now()})
    atomic_json(fold_root / "run_manifest.json", manifest)
    clean_scores = metric(saved_prediction, truth)
    for scenario_name in SCENARIOS:
        if scenario_name in completed:
            continue
        prediction = predict_model(model, scenarios[scenario_name], mean, std, device)
        scores = metric(prediction, truth)
        rows.append({
            "target_city": target, "seed": seed, "model": MODEL_NAME,
            "scenario": scenario_name, "rmse": scores["rmse"], "mae": scores["mae"],
            "p95_abs_error": scores["p95_abs_error"], "clean_rmse": clean_scores["rmse"],
            "clean_mae": clean_scores["mae"], "rmse_delta": scores["rmse"] - clean_scores["rmse"],
            "mae_delta": scores["mae"] - clean_scores["mae"],
            "rmse_relative_delta_pct": 100.0 * (scores["rmse"] / clean_scores["rmse"] - 1.0),
            "mae_relative_delta_pct": 100.0 * (scores["mae"] / clean_scores["mae"] - 1.0),
        })
        write_csv(partial_path, rows)
        completed.add(scenario_name)
        manifest.update({"completed_scenarios": sorted(completed), "completed_models": len(rows),
                         "updated_at_utc": utc_now()})
        atomic_json(fold_root / "run_manifest.json", manifest)

    write_csv(fold_root / "robustness_results.csv", rows)
    if partial_path.is_file():
        partial_path.unlink()
    full_rows = old["full_rows"]
    for row in full_rows:
        row["model"] = "full_scpcr"
    combined_rows = rows + full_rows
    write_csv(fold_root / "original_vs_full_results.csv", combined_rows)
    manifest.update({"perturbation_scores_generated": True, "completed_models": len(rows),
                     "result_row_count": len(rows), "full_reference_row_count": len(full_rows),
                     "combined_result_row_count": len(combined_rows), "status": "passed",
                     "stage": "complete", "finished_at_utc": utc_now(),
                     "updated_at_utc": utc_now()})
    atomic_json(fold_root / "run_manifest.json", manifest)
    (fold_root / "README.md").write_text(
        "# U2 Original PhaseFormer robustness\n\n"
        f"Status: `passed`; {target}/seed{seed}, six fixed perturbation scenarios.\n\n"
        "Original uses the existing canonical no-PCGrad checkpoint/predictions. The old P1 clean gate, fixed masks, and scenario protocol are read-only references. Full SCPRC rows are included in `original_vs_full_results.csv` as an unchanged reference copy.\n",
        encoding="utf-8",
    )
    return {"target_city": target, "seed": seed, "status": "passed", "stage": "complete",
            "output_root": str(fold_root), "completed_scenarios": sorted(completed),
            "result_row_count": len(rows)}


def aggregate(output_root: Path, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = []
    full_rows = []
    for target in CITIES:
        for seed in SEEDS:
            fold_root = output_root / f"target_{target.lower()}_seed{seed}"
            all_rows.extend(read_csv(fold_root / "robustness_results.csv"))
            full_rows.extend(read_csv(fold_root / "original_vs_full_results.csv"))
    write_csv(output_root / "original_results_all_folds.csv", all_rows)
    write_csv(output_root / "original_vs_full_all_folds.csv", full_rows)
    summaries = []
    for model in (MODEL_NAME, "full_scpcr"):
        for scenario in SCENARIOS:
            rows = [r for r in full_rows if r.get("model") == model and r.get("scenario") == scenario]
            def values(name: str) -> np.ndarray:
                return np.asarray([float(r[name]) for r in rows], dtype=np.float64)
            if not rows:
                continue
            rmses, maes = values("rmse"), values("mae")
            summaries.append({
                "model": model, "scenario": scenario, "fold_count": len(rows),
                "rmse_mean": float(rmses.mean()), "rmse_sd": float(rmses.std(ddof=1)),
                "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=1)),
                "clean_rmse_mean": float(values("clean_rmse").mean()),
                "clean_mae_mean": float(values("clean_mae").mean()),
                "rmse_delta_mean": float(values("rmse_delta").mean()),
                "mae_delta_mean": float(values("mae_delta").mean()),
            })
    summary_fields = list(summaries[0]) if summaries else ["model", "scenario", "fold_count"]
    with (output_root / "original_vs_full_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(summaries)
    status = "completed" if len(all_rows) == 18 * 6 and len(full_rows) == 18 * 12 else "incomplete_or_failed"
    payload = {
        "experiment": "U2 Original PhaseFormer vs Full SCPRC clean-gated robustness",
        "status": status, "generated_at_utc": utc_now(), "runner": str(HERE),
        "output_root": str(output_root), "old_reference_root": str(P1_ROOT),
        "cities": list(CITIES), "seeds": list(SEEDS), "scenario_names": list(SCENARIOS),
        "expected_original_rows": 18 * 6, "observed_original_rows": len(all_rows),
        "expected_combined_rows": 18 * 12, "observed_combined_rows": len(full_rows),
        "completed_fold_count": sum(r.get("status") == "passed" for r in fold_results),
        "fold_results": sorted(fold_results, key=lambda r: (r.get("target_city", ""), r.get("seed", 0))),
        "files": {
            "original_results_all_folds": str(output_root / "original_results_all_folds.csv"),
            "combined_results_all_folds": str(output_root / "original_vs_full_all_folds.csv"),
            "summary": str(output_root / "original_vs_full_summary.csv"),
        },
    }
    atomic_json(output_root / "multifold_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="run AMS/seed13 in smoke_validation only")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    output_root = args.output_root.resolve()
    if args.smoke:
        output_root = output_root / "smoke_validation"
        folds = [("AMS", 13)]
    else:
        folds = [(city, seed) for city in CITIES for seed in SEEDS]
    output_root.mkdir(parents=True, exist_ok=True)
    fold_results = []
    # A single GPU is kept as the default; workers can be raised explicitly.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_fold, city, seed, output_root=output_root,
                        device=args.device, torch_threads=args.torch_threads,
                        resume=not args.no_resume): (city, seed)
            for city, seed in folds
        }
        for future in as_completed(futures):
            result = future.result()
            fold_results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    if args.smoke:
        summary = {"experiment": "U2 smoke", "status": "passed" if all(
            r.get("status") == "passed" for r in fold_results
        ) else "blocked", "fold_results": fold_results,
                   "output_root": str(output_root), "generated_at_utc": utc_now()}
        atomic_json(output_root / "smoke_summary.json", summary)
        return 0 if summary["status"] == "passed" else 2
    payload = aggregate(output_root, fold_results)
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

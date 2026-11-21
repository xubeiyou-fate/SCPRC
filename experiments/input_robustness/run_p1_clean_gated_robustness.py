#!/usr/bin/env python3
"""Run a clean-gated single-fold CHARGED input robustness replay."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shlex
import socket
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
P1_ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'experiments/commands'
EXPERIMENTS = ROOT / 'scprc/training'
ARCHIVE = ROOT / "scprc/data_adapters/charged/Hourly.zip"
CANONICAL_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'
TARGET = "JHB"
SEED = 13
CONTEXT = 168
HORIZON = 24
TRAIN_END = 2928
VAL_END = 3672
MASK_SEED = 20261030
TORCH_THREADS = 2
DEVICE = "cuda"
MODELS = ("exact", "scpcr")
SCENARIO_NAMES = (
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

LEGACY_MODULE = "calibration_protocol_constants"
if LEGACY_MODULE not in sys.modules:
    legacy = type(sys)(LEGACY_MODULE)
    legacy.CONTEXT = protocol.CONTEXT
    legacy.HORIZON = protocol.HORIZON
    legacy.TRAIN_END = protocol.TRAIN_END
    legacy.protocol = protocol
    sys.modules[LEGACY_MODULE] = legacy

import scprc.training.train_scprc as pcgrad  # noqa: E402


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def file_sha256(path: Path) -> str:
    return digest(path, "sha256")


def json_digest(value: Any) -> str:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher = hashlib.sha256()
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(repr(array.shape).encode("utf-8"))
        hasher.update(array.view(np.uint8))
        return hasher.hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def required_output_names() -> list[str]:
    return [
        "clean_gate.json",
        "fixed_masks_and_noise.npz",
        "scenario_protocol.json",
        "robustness_results.csv",
        "run_manifest.json",
    ]


def output_is_complete(output_root: Path) -> bool:
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("status") == "passed"
        and manifest.get("perturbation_scores_generated") is True
        and all((output_root / name).is_file() for name in required_output_names())
    )


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RESULT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_partial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def mark_manifest_failure(output_root: Path, status: str, error: BaseException) -> None:
    path = output_root / "run_manifest.json"
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update({
            "status": status,
            "stage": "interrupted" if status == "interrupted" else "failed",
            "updated_at_utc": utc_now(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        })
        atomic_write_json(path, manifest)
    except (OSError, json.JSONDecodeError):
        return


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
    }


def load_fold() -> dict[str, Any]:
    fold = CANONICAL_ROOT / f"target_{TARGET.lower()}_seed{SEED}"
    result_path = fold / "result.json"
    prediction_path = fold / "predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing canonical fold artifacts under {fold}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {
            "label_starts",
            "truth_raw",
            "site_ids",
            "pred_exact_pcgrad_raw",
            "pred_scpcr_candidate_raw",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"canonical predictions missing {sorted(missing)}")
        reference = {key: archive[key].copy() for key in required}
    return {
        "fold": fold,
        "result_path": result_path,
        "prediction_path": prediction_path,
        "result": result,
        "reference": reference,
    }


def checkpoint_path(fold: Path, result: dict[str, Any], model_name: str) -> Path:
    """Resolve historical flat checkpoints and rebuilt nested checkpoints."""
    key = "exact_pcgrad" if model_name == "exact" else "scpcr_candidate"
    flat = fold / ("exact_pcgrad.pt" if model_name == "exact" else "scpcr_candidate.pt")
    if flat.is_file():
        return flat
    recorded = result.get("checkpoints", {}).get(key)
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    nested = fold / key / "checkpoint.pt"
    if nested.is_file():
        return nested
    return flat


def build_model(model_name: str, n_sites: int, result: dict[str, Any]) -> torch.nn.Module:
    if model_name == "exact":
        model = protocol._build_model("phaseformer_exact", n_sites)
    elif model_name == "scpcr":
        record = result.get("training", {}).get("scpcr_candidate", {})
        if not record:
            record = result.get("training", {}).get("full", {})
        threshold = record.get("source_relation_threshold", result.get("source_relation_threshold", 0.9))
        model = pcgrad.v21._build_model(n_sites, float(threshold), support_scale=1.0)
    else:
        raise ValueError(model_name)
    return model.float()


def predict(model: torch.nn.Module, histories: np.ndarray, mean: float, std: float,
            device: str) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(histories), 4):
            batch = (histories[offset:offset + 4] - mean) / std
            output = model(torch.from_numpy(batch).to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite forecast")
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32) * std + mean


def compare_metadata(cities: dict[str, dict[str, Any]], reference: dict[str, np.ndarray],
                     starts: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    city = cities[TARGET]
    site_ids = np.asarray(city["site_ids"], dtype=str)
    times = city["times"]
    timestamp_ok = bool(
        times[0].isoformat() == "2023-04-01T00:00:00"
        and times[-1].isoformat() == "2023-08-31T23:00:00"
        and all((times[index + 1] - times[index]).total_seconds() == 3600 for index in range(len(times) - 1))
    )
    return {
        "starts_equal": bool(np.array_equal(reference["label_starts"], starts)),
        "truth_equal": bool(np.array_equal(reference["truth_raw"], truth)),
        "site_order_equal": bool(np.array_equal(reference["site_ids"].astype(str), site_ids)),
        "timestamp_sequence_ok": timestamp_ok,
        "timestamp_first": str(times[0]),
        "timestamp_last": str(times[-1]),
        "site_count": int(len(site_ids)),
        "evaluation_origins": int(len(starts)),
        "truth_shape": list(truth.shape),
    }


def clean_gate(fold: dict[str, Any], cities: dict[str, dict[str, Any]],
               provenance: dict[str, Any], mean: float, std: float,
               starts: np.ndarray, truth: np.ndarray, histories: np.ndarray,
               device: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = fold["reference"]
    metadata = compare_metadata(cities, reference, starts, truth)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != TARGET)
    expected_provenance = fold["result"]["data_provenance"]
    scaler = {
        "source_cities": list(source_ids),
        "mean": float(mean),
        "std": float(std),
        "training_end_exclusive": TRAIN_END,
        "calculation": "equal-city-weight source mean and second moment over [0,2928)",
    }
    checks = {
        **metadata,
        "archive_md5_matches_published": digest(ARCHIVE, "md5") == protocol.ARCHIVE_MD5,
        "archive_size_matches_record": int(ARCHIVE.stat().st_size) == int(expected_provenance["archive_size_bytes"]),
        "city_prefix_hash_matches": provenance["city_prefix_sha256"] == expected_provenance["city_prefix_sha256"],
        "protocol_sha256_matches": digest(PROTOCOL_FILE) == fold["result"]["protocol_sha256"],
        "source_scaler_finite": bool(np.isfinite(mean) and np.isfinite(std) and std > 0),
    }
    predictions: dict[str, np.ndarray] = {}
    model_checks: dict[str, Any] = {}
    for model_name, checkpoint_name, reference_key in (
        ("exact", "exact_pcgrad.pt", "pred_exact_pcgrad_raw"),
        ("scpcr", "scpcr_candidate.pt", "pred_scpcr_candidate_raw"),
    ):
        checkpoint = checkpoint_path(fold["fold"], fold["result"], model_name)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(model_name, len(reference["site_ids"]), fold["result"])
        model.load_state_dict(checkpoint_data["state_dict"], strict=True)
        model.to(device)
        prediction = predict(model, histories, mean, std, device)
        predictions[model_name] = prediction
        reference_prediction = reference[reference_key]
        diff = prediction.astype(np.float64) - reference_prediction.astype(np.float64)
        scores = metric(prediction, truth)
        reference_scores = metric(reference_prediction, truth)
        model_checks[model_name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "checkpoint_model_name": checkpoint_data.get("model_name"),
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(np.isfinite(prediction).all()),
            "max_abs_difference_from_canonical": float(np.max(np.abs(diff))),
            "mean_abs_difference_from_canonical": float(np.mean(np.abs(diff))),
            "reproduced_metrics": scores,
            "canonical_metrics": reference_scores,
            "rmse_abs_difference": abs(scores["rmse"] - reference_scores["rmse"]),
            "mae_abs_difference": abs(scores["mae"] - reference_scores["mae"]),
        }
        checks[f"{model_name}_shape_matches"] = bool(prediction.shape == reference_prediction.shape)
        checks[f"{model_name}_finite"] = bool(np.isfinite(prediction).all())
        checks[f"{model_name}_prediction_match"] = bool(np.max(np.abs(diff)) <= 1e-5)
        checks[f"{model_name}_rmse_match"] = bool(abs(scores["rmse"] - reference_scores["rmse"]) <= 1e-6)
        checks[f"{model_name}_mae_match"] = bool(abs(scores["mae"] - reference_scores["mae"]) <= 1e-6)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks["all_metadata_checks"] = all(
        value for key, value in checks.items()
        if key not in {"all_metadata_checks", "gate_pass"}
        and isinstance(value, bool)
    )
    checks["gate_pass"] = bool(checks["all_metadata_checks"])
    return {
        "status": "passed" if checks["gate_pass"] else "blocked",
        "checks": checks,
        "scaler": scaler,
        "models": model_checks,
    }, predictions


def make_scenarios(histories: np.ndarray, mean: float, std: float) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(MASK_SEED)
    shape = histories.shape
    n_origins, _, n_sites = shape
    scenarios: dict[str, np.ndarray] = {}
    descriptions: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}

    point_mask = rng.random(shape) < 0.10
    point_data = histories.copy()
    point_data[point_mask] = mean
    scenarios["point_missing_10pct_mean_impute"] = point_data
    saved["point_missing_mask"] = point_mask
    descriptions["point_missing_10pct_mean_impute"] = {
        "kind": "missing",
        "replacement": "source_scaler_mean",
        "mask_fraction": float(point_mask.mean()),
        "mask_sha256": json_digest(point_mask.astype(np.uint8)),
    }

    block_starts = rng.integers(0, CONTEXT - 24 + 1, size=n_origins, dtype=np.int64)
    block_mask = np.zeros(shape, dtype=bool)
    for index, start in enumerate(block_starts):
        block_mask[index, int(start):int(start) + 24, :] = True
    block_data = histories.copy()
    block_data[block_mask] = mean
    scenarios["temporal_block_missing_24h_mean_impute"] = block_data
    saved["temporal_block_mask"] = block_mask
    saved["temporal_block_starts"] = block_starts
    descriptions["temporal_block_missing_24h_mean_impute"] = {
        "kind": "missing_contiguous_block",
        "block_hours": 24,
        "replacement": "source_scaler_mean",
        "mask_fraction": float(block_mask.mean()),
        "block_starts_sha256": json_digest(block_starts),
    }

    outage_count = max(1, int(np.ceil(0.10 * n_sites)))
    outage_sites = np.sort(rng.choice(n_sites, size=outage_count, replace=False)).astype(np.int64)
    site_mask = np.zeros(shape, dtype=bool)
    site_mask[:, :, outage_sites] = True
    site_data = histories.copy()
    site_data[site_mask] = mean
    scenarios["site_outage_10pct_mean_impute"] = site_data
    saved["site_outage_mask"] = site_mask
    saved["site_outage_indices"] = outage_sites
    descriptions["site_outage_10pct_mean_impute"] = {
        "kind": "site_outage",
        "replacement": "source_scaler_mean",
        "site_count": int(outage_count),
        "site_fraction": float(outage_count / n_sites),
        "site_indices": outage_sites.tolist(),
    }

    scenarios["scale_shift_plus20pct"] = np.maximum(histories * 1.20, 0.0).astype(np.float32)
    descriptions["scale_shift_plus20pct"] = {
        "kind": "multiplicative_input_scale_shift",
        "factor": 1.20,
        "clip": "lower_bound_zero",
    }

    noise = rng.normal(0.0, 0.10 * std, size=shape).astype(np.float32)
    scenarios["gaussian_noise_sigma10pct_source_std"] = np.maximum(histories + noise, 0.0).astype(np.float32)
    saved["gaussian_noise"] = noise
    descriptions["gaussian_noise_sigma10pct_source_std"] = {
        "kind": "additive_gaussian_noise",
        "sigma": float(0.10 * std),
        "sigma_as_source_std_fraction": 0.10,
        "clip": "lower_bound_zero",
        "noise_sha256": json_digest(noise),
    }

    spike_mask = rng.random(shape) < 0.01
    spike_data = histories.copy()
    spike_data[spike_mask] = spike_data[spike_mask] * 3.0
    scenarios["positive_spike_1pct_x3"] = spike_data
    saved["spike_mask"] = spike_mask
    descriptions["positive_spike_1pct_x3"] = {
        "kind": "positive_spike",
        "mask_fraction": float(spike_mask.mean()),
        "amplitude_factor": 3.0,
        "mask_sha256": json_digest(spike_mask.astype(np.uint8)),
    }
    return scenarios, descriptions, saved


def write_blocker(reason: str, payload: dict[str, Any]) -> None:
    (P1_ROOT / "blocker_report.md").write_text(
        "# P1 Clean-Gated Robustness Blocker\n\n"
        "Status: `blocked`; no corrupted-input scores were generated.\n\n"
        f"Reason: {reason}\n\n"
        "## Required repairs\n\n"
        "- Restore a hash-matched model/checkpoint/code identity chain.\n"
        "- Restore exact input-history, timestamp, station-order, and scaler provenance.\n"
        "- Re-run this single-fold clean gate and require every recorded check to pass.\n"
        "- Only after the gate passes, generate identical fixed masks for Exact and Full SCPRC.\n\n"
        "## Gate payload\n\n```json\n"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def run(device: str, *, target: str = TARGET, seed: int = SEED,
        output_root: Path = P1_ROOT, mask_seed: int = MASK_SEED,
        torch_threads: int = TORCH_THREADS, resume: bool = True,
        canonical_root: Path = CANONICAL_ROOT) -> int:
    """Replay one fold without changing the canonical checkpoint artifacts.

    The original JHB/seed13 invocation remains the default.  The explicit
    fold/output arguments make the clean-gated replay reusable for the other
    17 city-seed folds while keeping every fold in its own directory.
    """
    global TARGET, SEED, P1_ROOT, MASK_SEED, TORCH_THREADS, CANONICAL_ROOT
    TARGET = str(target).upper()
    SEED = int(seed)
    P1_ROOT = Path(output_root)
    MASK_SEED = int(mask_seed)
    TORCH_THREADS = int(torch_threads)
    CANONICAL_ROOT = Path(canonical_root).resolve()
    P1_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = P1_ROOT / "run_manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_manifest = {}
    if resume and output_is_complete(P1_ROOT):
        return 0

    attempt_number = int(previous_manifest.get("attempt_number", 0)) + 1
    manifest: dict[str, Any] = {
        "experiment": "P1 clean-gated CHARGED input robustness",
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "status": "running",
        "stage": "clean_gate",
        "fold_id": f"{TARGET.lower()}_seed{SEED}",
        "target_city": TARGET,
        "seed": SEED,
        "device": device,
        "torch_threads": TORCH_THREADS,
        "archive": str(ARCHIVE),
        "canonical_root": str(CANONICAL_ROOT),
        "canonical_fold": str(CANONICAL_ROOT / f"target_{TARGET.lower()}_seed{SEED}"),
        "protocol_file": str(PROTOCOL_FILE),
        "context_hours": CONTEXT,
        "horizon_hours": HORIZON,
        "train_end_exclusive": TRAIN_END,
        "evaluation_window": "2023-08-01 through 2023-08-31",
        "mask_seed": MASK_SEED,
        "models": list(MODELS),
        "scenario_names": list(SCENARIO_NAMES),
        "required_outputs": required_output_names(),
        "output_root": str(P1_ROOT),
        "working_directory": str(ROOT),
        "command": shlex.join(sys.argv),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "attempt_id": f"{TARGET.lower()}_seed{SEED}_attempt{attempt_number}",
        "attempt_number": attempt_number,
        "resume_requested": bool(resume),
        "resumed_from_status": previous_manifest.get("status"),
        "resumable": True,
        "resume_policy": (
            "reuse a passed output; otherwise retain completed scenarios in "
            "robustness_results.partial.csv and continue the missing scenarios"
        ),
        "partial_results_file": str(P1_ROOT / "robustness_results.partial.csv"),
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "completed_scenarios": [],
        "scenario_count": len(SCENARIO_NAMES),
        "perturbation_scores_generated": False,
    }
    atomic_write_json(manifest_path, manifest)
    torch.set_num_threads(TORCH_THREADS)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    fold = load_fold()
    cities, provenance = protocol.load_cities(ARCHIVE)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != TARGET)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, values = protocol._validation_data(cities[TARGET])
    histories = np.stack([values[start - CONTEXT:start] for start in starts]).astype(np.float32)
    gate, clean_predictions = clean_gate(
        fold, cities, provenance, mean, std, starts, truth, histories, device,
    )
    manifest.update({
        "status": gate["status"],
        "stage": "clean_gate_passed" if gate["status"] == "passed" else "clean_gate_failed",
        "archive_sha256": digest(ARCHIVE),
        "archive_md5": digest(ARCHIVE, "md5"),
        "canonical_fold": str(fold["fold"]),
        "canonical_result_sha256": digest(fold["result_path"]),
        "canonical_predictions_sha256": digest(fold["prediction_path"]),
        "protocol_sha256": digest(PROTOCOL_FILE),
        "validation_origins": int(len(starts)),
        "source_cities": list(source_ids),
        "gate": gate,
        "updated_at_utc": utc_now(),
    })
    atomic_write_json(P1_ROOT / "clean_gate.json", gate)
    atomic_write_json(manifest_path, manifest)
    if not gate["checks"]["gate_pass"]:
        write_blocker("one or more clean scprc checks failed", gate)
        manifest["perturbation_scores_generated"] = False
        atomic_write_json(manifest_path, manifest)
        return 2

    scenarios, descriptions, saved_masks = make_scenarios(histories, mean, std)
    fixed_mask_path = P1_ROOT / "fixed_masks_and_noise.npz"
    if not (resume and fixed_mask_path.is_file()):
        np.savez_compressed(fixed_mask_path, **saved_masks)
    scenario_protocol_path = P1_ROOT / "scenario_protocol.json"
    atomic_write_json(scenario_protocol_path, descriptions)

    partial_path = P1_ROOT / "robustness_results.partial.csv"
    rows = read_partial_rows(partial_path) if resume else []
    completed_counts = {
        scenario: sum(row.get("scenario") == scenario for row in rows)
        for scenario in scenarios
    }
    complete_scenarios = {
        scenario for scenario, count in completed_counts.items()
        if count == len(MODELS)
    }
    rows = [row for row in rows if row.get("scenario") in complete_scenarios]
    manifest.update({
        "stage": "scores",
        "status": "running",
        "scenarios": list(scenarios),
        "fixed_mask_file": str(fixed_mask_path),
        "scenario_protocol_file": str(scenario_protocol_path),
        "completed_scenarios": sorted(complete_scenarios),
        "updated_at_utc": utc_now(),
    })
    write_results_csv(partial_path, rows)
    atomic_write_json(manifest_path, manifest)
    for scenario_name, scenario_histories in scenarios.items():
        if scenario_name in complete_scenarios:
            continue
        for model_name in MODELS:
            model_path = checkpoint_path(fold["fold"], fold["result"], model_name)
            checkpoint_data = torch.load(model_path, map_location="cpu", weights_only=False)
            model = build_model(model_name, histories.shape[2], fold["result"])
            model.load_state_dict(checkpoint_data["state_dict"], strict=True)
            model.to(device)
            prediction = predict(model, scenario_histories, mean, std, device)
            scores = metric(prediction, truth)
            clean_scores = metric(clean_predictions[model_name], truth)
            rows.append({
                "target_city": TARGET,
                "seed": SEED,
                "model": model_name,
                "scenario": scenario_name,
                "rmse": scores["rmse"],
                "mae": scores["mae"],
                "p95_abs_error": scores["p95_abs_error"],
                "clean_rmse": clean_scores["rmse"],
                "clean_mae": clean_scores["mae"],
                "rmse_delta": scores["rmse"] - clean_scores["rmse"],
                "mae_delta": scores["mae"] - clean_scores["mae"],
                "rmse_relative_delta_pct": 100.0 * (scores["rmse"] / clean_scores["rmse"] - 1.0),
                "mae_relative_delta_pct": 100.0 * (scores["mae"] / clean_scores["mae"] - 1.0),
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_results_csv(partial_path, rows)
        complete_scenarios.add(scenario_name)
        manifest.update({
            "completed_scenarios": sorted(complete_scenarios),
            "completed_models": len(rows),
            "updated_at_utc": utc_now(),
        })
        atomic_write_json(manifest_path, manifest)

    write_results_csv(P1_ROOT / "robustness_results.csv", rows)
    if partial_path.is_file():
        partial_path.unlink()
    manifest["perturbation_scores_generated"] = True
    manifest["scenarios"] = list(scenarios)
    manifest["fixed_mask_file"] = str(fixed_mask_path)
    manifest["results_file"] = str(P1_ROOT / "robustness_results.csv")
    manifest["result_row_count"] = len(rows)
    manifest["scenario_count"] = len(scenarios)
    manifest["completed_models"] = len(rows)
    manifest["status"] = "passed"
    manifest["stage"] = "complete"
    manifest["finished_at_utc"] = utc_now()
    manifest["updated_at_utc"] = manifest["finished_at_utc"]
    atomic_write_json(manifest_path, manifest)
    (P1_ROOT / "README.md").write_text(
        "# P1 Clean-Gated Input Robustness\n\n"
        f"Status: `clean gate passed`; fixed-mask perturbation replay completed for {TARGET}/seed{SEED}.\n\n"
        "The replay uses the canonical homogeneous fold, the local CHARGED `Hourly.zip`, source-only scaler, and the saved Exact/SCPRC checkpoints. Exact and Full SCPRC receive identical fixed perturbations per scenario.\n\n"
        "Artifacts: `clean_gate.json`, `run_manifest.json`, `scenario_protocol.json`, `fixed_masks_and_noise.npz`, and `robustness_results.csv`.\n\n"
        "This is one retrospective city-seed fold, not a six-city robustness estimate. The multifold summary is the only entry point for a six-city aggregate.\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--target", choices=protocol.CITY_CODES, default=TARGET)
    parser.add_argument("--seed", type=int, choices=(13, 17, 29), default=SEED)
    parser.add_argument("--output-root", type=Path, default=P1_ROOT)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--mask-seed", type=int, default=MASK_SEED)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="reuse a completed output or continue partial scenario results")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="start a fresh attempt in the supplied output directory")
    args = parser.parse_args()
    try:
        return run(args.device, target=args.target, seed=args.seed,
                   output_root=args.output_root, mask_seed=args.mask_seed,
                   torch_threads=args.torch_threads, resume=args.resume,
                   canonical_root=args.canonical_root)
    except KeyboardInterrupt as error:
        mark_manifest_failure(args.output_root, "interrupted", error)
        raise
    except Exception as error:
        mark_manifest_failure(args.output_root, "failed", error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

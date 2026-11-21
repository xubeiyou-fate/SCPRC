#!/usr/bin/env python3
"""Retrain one canonical h=24 anchor fold with four independent PCGrad arms.

Every arm is constructed locally and trained through the canonical
``_train_candidate_pcgrad`` function.  The runner never reads a previous
control or candidate checkpoint.  A fold may be resumed after an interrupted
process: an arm is considered reusable only when its own checkpoint,
prediction archive, and result record all pass the artifact checks below.

The default full-budget invocation is 2,048 source minibatches.  Smaller
budgets are allowed for an explicit smoke run and are written to a separate
output root by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve()
ROOT = next(
    parent for parent in HERE.parents
    if (parent / 'scprc/training').is_dir()
)
EXPERIMENTS = ROOT / 'scprc/training'
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'
OUTPUT_DEFAULT = ROOT / 'experiments/forecasting/h24_framework'
PROTOCOL_ID = "scpcr-matched-baseline-20261023"
FULL_STEPS = 2048
SEEDS = (13, 17, 29)

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The canonical source adapter keeps this historical import for a module that
# is not used by its PCGrad training path.  Keep the compatibility shim local
# to this new runner; no old experiment output is imported.
_legacy_name = "calibration_protocol_constants"
if _legacy_name not in sys.modules:
    _legacy = types.ModuleType(_legacy_name)
    _legacy.CONTEXT = 168
    _legacy.HORIZON = 24
    _legacy.TRAIN_END = 2928
    sys.modules[_legacy_name] = _legacy

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.train_scprc as pcgrad  # noqa: E402


ARMS: dict[str, dict[str, str]] = {
    "exact_pcgrad": {
        "model_name": "phaseformer_exact",
        "description": "PhaseFormer Exact trained with source-city canonical PCGrad",
    },
    "daily_relation_pcgrad": {
        "model_name": "phase_agnostic_daily",
        "description": "daily relation adapter trained with source-city canonical PCGrad",
    },
    "phase_relation_pcgrad": {
        "model_name": "phase_relation",
        "description": "phase relation adapter trained with source-city canonical PCGrad",
    },
    "scpcr_candidate": {
        "model_name": "scprc",
        "description": "full SCPRC relation adapter trained with source-city canonical PCGrad",
    },
}


def project_artifact_path(value):
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scprc.project_paths import project_artifact_path as resolve
    return resolve(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model(arm: str, n_sites: int, threshold: float) -> torch.nn.Module:
    """Build the requested arm without consulting any prior artifact."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected {tuple(ARMS)}")
    if arm == "scpcr_candidate":
        # This is the same source-certified architecture used by the
        # canonical SCPRC implementation, with support projection fixed to 1.
        return pcgrad.v21._build_model(
            n_sites, float(threshold), support_scale=1.0
        ).float()
    return protocol._build_model(ARMS[arm]["model_name"], n_sites).float()


def train_arm(
    arm: str,
    cities: dict[str, dict[str, Any]],
    source_ids: tuple[str, ...],
    n_sites: int,
    mean: float,
    std: float,
    *,
    seed: int,
    steps: int,
    threshold: float,
    device: str,
    torch_threads: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct and train one arm using the canonical PCGrad function."""
    seed_everything(seed)
    model = build_model(arm, n_sites, threshold).to(device)

    # All four arms, including Exact and Full SCPRC, pass through this one
    # canonical source-city PCGrad implementation.
    info = pcgrad._train_candidate_pcgrad(
        model,
        cities,
        source_ids,
        mean,
        std,
        seed=seed,
        steps=steps,
        device=device,
        torch_threads=torch_threads,
    )
    info = dict(info)
    info.update({
        "arm": arm,
        "model": ARMS[arm]["model_name"],
        "protocol_id": PROTOCOL_ID,
        "canonical_training_function": (
            "train_scprc._train_candidate_pcgrad"
        ),
        "reference_type": "new_independent_retraining",
    })
    if arm == "scpcr_candidate":
        info.update({
            "source_relation_threshold": float(threshold),
            "source_relation_threshold_quantile": 0.90,
            "support_scale": 1.0,
            "target_support_projection": False,
            "target_release_gate": False,
        })
    return model, info


def arm_paths(fold_dir: Path, arm: str) -> dict[str, Path]:
    arm_dir = fold_dir / arm
    return {
        "directory": arm_dir,
        "checkpoint": arm_dir / "checkpoint.pt",
        "predictions": arm_dir / "predictions.npz",
        "result": arm_dir / "result.json",
    }


def load_complete_arm(
    fold_dir: Path,
    arm: str,
    *,
    expected_steps: int,
) -> tuple[dict[str, Any], np.ndarray] | None:
    paths = arm_paths(fold_dir, arm)
    if not all(paths[key].is_file() for key in ("checkpoint", "predictions", "result")):
        return None
    try:
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        if result.get("status") != ("completed" if expected_steps == FULL_STEPS else "completed_smoke"):
            return None
        if int(result.get("steps", -1)) != expected_steps:
            return None
        with np.load(paths["predictions"], allow_pickle=False) as archive:
            if "prediction_raw" not in archive.files:
                return None
            prediction = archive["prediction_raw"].copy()
        if prediction.ndim != 3 or not np.isfinite(prediction).all():
            return None
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint.get("arm") != arm or "state_dict" not in checkpoint:
            return None
    except (OSError, ValueError, KeyError, RuntimeError, TypeError):
        return None
    return result, prediction


def write_arm_artifacts(
    fold_dir: Path,
    arm: str,
    model: torch.nn.Module,
    info: dict[str, Any],
    prediction: np.ndarray,
    starts: np.ndarray,
    truth: np.ndarray,
    site_ids: np.ndarray,
    *,
    target: str,
    seed: int,
    steps: int,
    threshold: float,
    protocol_sha256: str,
) -> dict[str, Any]:
    paths = arm_paths(fold_dir, arm)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "arm": arm,
        "model_name": ARMS[arm]["model_name"],
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "source_relation_threshold": float(threshold) if arm == "scpcr_candidate" else None,
        "support_scale": 1.0 if arm == "scpcr_candidate" else None,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    atomic_torch_save(paths["checkpoint"], checkpoint_payload)
    atomic_save_npz(
        paths["predictions"],
        label_starts=starts,
        truth_raw=truth,
        site_ids=site_ids,
        prediction_raw=prediction.astype(np.float32),
        **{f"pred_{arm}_raw": prediction.astype(np.float32)},
    )
    status = "completed" if steps == FULL_STEPS else "completed_smoke"
    arm_result = {
        "experiment": "canonical h24 anchor four-arm PCGrad retraining",
        "status": status,
        "smoke": steps != FULL_STEPS,
        "target_city": target,
        "seed": seed,
        "arm": arm,
        "description": ARMS[arm]["description"],
        "model_name": ARMS[arm]["model_name"],
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "steps": steps,
        "training": info,
        "metrics": protocol._score(prediction, truth),
        "checkpoint": str(paths["checkpoint"]),
        "predictions": str(paths["predictions"]),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "predictions_sha256": sha256(paths["predictions"]),
        "reference_type": "new_independent_retraining",
    }
    atomic_write_json(paths["result"], arm_result)
    return arm_result


def fold_complete(fold_dir: Path, *, steps: int = FULL_STEPS) -> bool:
    expected_status = "completed" if steps == FULL_STEPS else "completed_smoke"
    result_path = fold_dir / "result.json"
    predictions_path = fold_dir / "predictions.npz"
    if not result_path.is_file() or not predictions_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != expected_status or int(result.get("steps", -1)) != steps:
            return False
        if tuple(result.get("arms", ())) != tuple(ARMS):
            return False
        with np.load(predictions_path, allow_pickle=False) as archive:
            required = {"label_starts", "truth_raw", "site_ids"}
            required.update(f"pred_{arm}_raw" for arm in ARMS)
            if not required.issubset(archive.files):
                return False
            numeric_keys = required - {"site_ids"}
            if not all(np.isfinite(archive[key]).all() for key in numeric_keys):
                return False
        return all(load_complete_arm(fold_dir, arm, expected_steps=steps) is not None for arm in ARMS)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_fold(
    target: str,
    seed: int,
    *,
    output_root: Path = OUTPUT_DEFAULT,
    steps: int = FULL_STEPS,
    device: str = "cuda",
    torch_threads: int = 2,
    archive_path: Path = protocol.ARCHIVE,
    resume: bool = False,
) -> dict[str, Any]:
    target = target.upper()
    if target not in protocol.CITY_CODES:
        raise ValueError(f"invalid target {target}; expected {protocol.CITY_CODES}")
    if seed not in SEEDS:
        raise ValueError(f"invalid seed {seed}; expected {SEEDS}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_root = output_root.resolve()
    fold_dir = output_root / f"target_{target.lower()}_seed{seed}"
    result_path = fold_dir / "result.json"
    if fold_complete(fold_dir, steps=steps):
        return json.loads(result_path.read_text(encoding="utf-8"))
    if result_path.exists() and not resume:
        raise FileExistsError(
            f"incomplete or incompatible fold exists at {fold_dir}; use --resume to rebuild it"
        )
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))

    protocol_digest = sha256(PROTOCOL_FILE)
    cities, data_provenance = protocol.load_cities(archive_path)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    site_ids = np.asarray(cities[target]["site_ids"], dtype=str)
    threshold = pcgrad.v21.cert._estimate_threshold(
        cities,
        source_ids,
        mean,
        std,
        seed=seed,
        device=device,
        quantile=0.90,
    )
    training: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    predictions: dict[str, np.ndarray] = {
        "label_starts": starts,
        "truth_raw": truth,
        "site_ids": site_ids,
    }
    progress = {
        "target_city": target,
        "seed": seed,
        "steps": steps,
        "status": "running",
        "arms": {arm: "pending" for arm in ARMS},
        "updated_at_unix": time.time(),
    }
    atomic_write_json(fold_dir / "progress.json", progress)
    started = time.perf_counter()

    for arm in ARMS:
        cached = load_complete_arm(fold_dir, arm, expected_steps=steps) if resume else None
        if cached is not None:
            arm_result, prediction = cached
            progress["arms"][arm] = "resumed"
        else:
            progress["arms"][arm] = "training"
            progress["updated_at_unix"] = time.time()
            atomic_write_json(fold_dir / "progress.json", progress)
            model, info = train_arm(
                arm,
                cities,
                source_ids,
                len(site_ids),
                mean,
                std,
                seed=seed,
                steps=steps,
                threshold=float(threshold),
                device=device,
                torch_threads=torch_threads,
            )
            prediction = protocol._predict(
                model,
                target_values,
                starts,
                mean,
                std,
                device=device,
                batch_size=4,
            ).astype(np.float32)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid prediction for {arm}: {prediction.shape}")
            arm_result = write_arm_artifacts(
                fold_dir,
                arm,
                model,
                info,
                prediction,
                starts,
                truth,
                site_ids,
                target=target,
                seed=seed,
                steps=steps,
                threshold=float(threshold),
                protocol_sha256=protocol_digest,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress["arms"][arm] = "completed"
        training[arm] = arm_result["training"]
        metrics[arm] = arm_result["metrics"]
        checkpoints[arm] = arm_result["checkpoint"]
        predictions[f"pred_{arm}_raw"] = prediction
        progress["updated_at_unix"] = time.time()
        atomic_write_json(fold_dir / "progress.json", progress)

    status = "completed" if steps == FULL_STEPS else "completed_smoke"
    atomic_save_npz(fold_dir / "predictions.npz", **predictions)
    result = {
        "experiment": "canonical h24 anchor four-arm PCGrad retraining",
        "status": status,
        "smoke": steps != FULL_STEPS,
        "target_city": target,
        "seed": seed,
        "source_cities": list(source_ids),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_digest,
        "protocol_file": str(PROTOCOL_FILE),
        "steps": steps,
        "arms": list(ARMS),
        "arm_definitions": ARMS,
        "training": training,
        "metrics": metrics,
        "checkpoints": checkpoints,
        "predictions": str(fold_dir / "predictions.npz"),
        "prediction_sha256": sha256(fold_dir / "predictions.npz"),
        "independent_retraining": True,
        "old_artifacts_read": False,
        "canonical_training_function": (
            "train_scprc._train_candidate_pcgrad"
        ),
        "data_provenance": data_provenance,
        "source_scaler": {"mean": mean, "std": std},
        "source_relation_threshold": float(threshold),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(
        fold_dir / "protocol.json",
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_digest,
            "source_minibatches": steps,
            "context_hours": protocol.CONTEXT,
            "horizon_hours": protocol.HORIZON,
            "arms": list(ARMS),
            "independent_retraining": True,
            "old_artifacts_read": False,
        },
    )
    atomic_write_json(
        fold_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_requested": device,
            "torch_threads": torch_threads,
            "pid": os.getpid(),
        },
    )
    code_paths = (
        HERE,
        PROTOCOL_FILE,
        protocol.HERE,
        pcgrad.HERE,
        pcgrad.v21.HERE,
        EXPERIMENTS / "phase_relations.py",
        ROOT / "scprc/models/scprc.py",
        ROOT / "scprc/baselines/phaseformer_adapter.py",
    )
    result["code_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in code_paths
        if path.is_file()
    }
    atomic_write_json(fold_dir / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    atomic_write_json(fold_dir / "result.json", result)
    progress.update({"status": status, "updated_at_unix": time.time()})
    atomic_write_json(fold_dir / "progress.json", progress)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=protocol.CITY_CODES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--archive", type=Path, default=protocol.ARCHIVE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_fold(
        args.target,
        args.seed,
        output_root=args.output_root,
        steps=args.steps,
        device=args.device,
        torch_threads=args.torch_threads,
        archive_path=args.archive,
        resume=args.resume,
    )
    print(json.dumps({
        "status": result["status"],
        "target": result["target_city"],
        "seed": result["seed"],
        "steps": result["steps"],
        "metrics": result["metrics"],
        "output": str(project_artifact_path(result["predictions"]).parent),
    }, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

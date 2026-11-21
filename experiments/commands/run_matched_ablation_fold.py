#!/usr/bin/env python3
"""Run one current-protocol SCPRC offline ablation fold.

The three trained arms are Exact-PCGrad, daily-relation-PCGrad, and the
uncertified phase-relation-PCGrad control.  The full SCPRC candidate is loaded
from the frozen current candidate matrix and remains an offline raw forecast;
Online release and support/fallback decisions are never scored here.
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

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'
PROTOCOL_ID = "scpcr-matched-baseline-20261023"
OUTPUT_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261026"
CURRENT_CANDIDATE_ROOT = ROOT / "experiments/supplementary/current_scpcr_charged_20261024"
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402

# The source-certificate module retains an unused historical import.  Mirror
# the compatibility shim used by the current frozen candidate runner without
# importing any legacy training or release logic.
_legacy_name = "calibration_protocol_constants"
if _legacy_name not in sys.modules:
    _legacy = types.ModuleType(_legacy_name)
    _legacy.CONTEXT = protocol.CONTEXT
    _legacy.HORIZON = protocol.HORIZON
    _legacy.TRAIN_END = protocol.TRAIN_END
    _legacy.protocol = protocol
    sys.modules[_legacy_name] = _legacy

import scprc.training.train_scprc as pcgrad  # noqa: E402


ARMS = (
    "exact_pcgrad",
    "daily_relation_pcgrad",
    "phase_relation_pcgrad",
)
MODEL_NAMES = {
    "exact_pcgrad": "phaseformer_exact",
    "daily_relation_pcgrad": "phase_agnostic_daily",
    "phase_relation_pcgrad": "phase_relation",
}


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


def arm_model(name: str, n_sites: int) -> torch.nn.Module:
    return protocol._build_model(MODEL_NAMES[name], n_sites)


def load_candidate_reference(target: str, seed: int) -> tuple[dict, dict, Path]:
    source_dir = CURRENT_CANDIDATE_ROOT / f"target_{target.lower()}_seed{seed}"
    result_path = source_dir / "result.json"
    prediction_path = source_dir / "predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing current candidate reference for {target}/{seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol_id = result.get("current_run_provenance", {}).get("protocol_id")
    if protocol_id != "scpcr-matched-protocol-20261023":
        raise ValueError(f"candidate reference has wrong protocol: {protocol_id}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {
            "label_starts", "truth_raw", "site_ids",
            "pred_source_certified_internal_candidate_raw",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"candidate reference missing {sorted(missing)}")
        candidate = {
            key: archive[key].copy()
            for key in required
        }
    if candidate["truth_raw"].shape != (
        len(candidate["label_starts"]), 24, len(candidate["site_ids"])
    ):
        raise ValueError("candidate reference has unexpected forecast shape")
    return result, candidate, source_dir


def train_arm(name: str, cities: dict, source_ids: tuple[str, ...],
              target_sites: int, mean: float, std: float, *, seed: int,
              steps: int, device: str, torch_threads: int) -> tuple[torch.nn.Module, dict]:
    seed_everything(seed)
    model = arm_model(name, target_sites).float().to(device)
    info = pcgrad._train_candidate_pcgrad(
        model, cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads,
    )
    info = dict(info)
    info.update({
        "arm": name,
        "model": MODEL_NAMES[name],
        "gradient_aggregation": "PCGrad pairwise projection on conflicting source-city gradients",
        "protocol_id": PROTOCOL_ID,
    })
    return model, info


def train_offline_full_candidate(cities: dict, source_ids: tuple[str, ...],
                                 target_sites: int, mean: float, std: float,
                                 *, seed: int, steps: int, device: str,
                                 torch_threads: int) -> tuple[torch.nn.Module, dict]:
    """Train full SCPRC without target-support projection or release gating."""
    seed_everything(seed)
    threshold = pcgrad.v21.cert._estimate_threshold(
        cities, source_ids, mean, std, seed=seed, device=device, quantile=0.90,
    )
    model = pcgrad.v21._build_model(target_sites, threshold, support_scale=1.0)
    model = model.float().to(device)
    info = pcgrad._train_candidate_pcgrad(
        model, cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads,
    )
    info = dict(info)
    info.update({
        "arm": "scpcr_candidate",
        "model": "scprc",
        "source_relation_threshold": threshold,
        "support_scale": 1.0,
        "target_support_projection": False,
        "target_release_gate": False,
        "protocol_id": PROTOCOL_ID,
    })
    return model, info


def run_fold(target: str, seed: int, *, output_root: Path = OUTPUT_ROOT,
             steps: int = 2048, device: str = "cuda",
             torch_threads: int = 2,
             archive_path: Path = protocol.ARCHIVE) -> dict:
    target = target.upper()
    if target not in protocol.CITY_CODES:
        raise ValueError(f"invalid target {target}")
    if seed not in (13, 17, 29):
        raise ValueError("the frozen matrix uses seeds 13, 17, and 29")
    if steps != 2048:
        raise ValueError("the frozen matrix requires exactly 2048 source minibatches")
    out = output_root / f"target_{target.lower()}_seed{seed}"
    result_path = out / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    out.mkdir(parents=True, exist_ok=True)
    cities, data_provenance = protocol.load_cities(archive_path)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    n_sites = len(cities[target]["site_ids"])
    predictions = {
        "label_starts": starts,
        "truth_raw": truth,
        "site_ids": np.asarray(cities[target]["site_ids"], dtype=str),
    }
    training = {}
    checkpoints = {}
    started = time.perf_counter()
    for arm in ARMS:
        model, info = train_arm(
            arm, cities, source_ids, n_sites, mean, std, seed=seed,
            steps=steps, device=device, torch_threads=torch_threads,
        )
        prediction = protocol._predict(
            model, target_values, starts, mean, std, device=device, batch_size=4,
        ).astype(np.float32)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"unexpected prediction for {arm}: {prediction.shape}")
        predictions[f"pred_{arm}_raw"] = prediction
        training[arm] = {**info, "validation_raw_volume": protocol._score(prediction, truth)}
        checkpoint_path = out / f"{arm}.pt"
        torch.save({
            "arm": arm,
            "model_name": MODEL_NAMES[arm],
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, checkpoint_path)
        checkpoints[arm] = checkpoint_path
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_result, candidate, candidate_dir = load_candidate_reference(target, seed)
    candidate_skipped = bool(candidate_result.get("candidate", {}).get("skipped", False))
    candidate_checkpoint = candidate_dir / "scpcr_candidate.pt"
    if candidate_skipped:
        candidate_model, candidate_info = train_offline_full_candidate(
            cities, source_ids, n_sites, mean, std, seed=seed, steps=steps,
            device=device, torch_threads=torch_threads,
        )
        candidate_prediction = protocol._predict(
            candidate_model, target_values, starts, mean, std, device=device, batch_size=4,
        ).astype(np.float32)
        candidate_checkpoint = out / "scpcr_candidate.pt"
        torch.save({
            "arm": "scpcr_candidate",
            "model_name": "scprc",
            "support_scale": 1.0,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in candidate_model.state_dict().items()
            },
        }, candidate_checkpoint)
        training["scpcr_candidate"] = {
            **candidate_info,
            "validation_raw_volume": protocol._score(candidate_prediction, truth),
            "reference_type": "new_offline_no_support_projection_training",
        }
        del candidate_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        candidate_prediction = candidate["pred_source_certified_internal_candidate_raw"].astype(np.float32)
        training["scpcr_candidate"] = {
            "reference_type": "frozen_current_raw_candidate",
            "validation_raw_volume": protocol._score(candidate_prediction, truth),
            "parameters": candidate_result.get("candidate", {}).get("parameters"),
            "support_projection": False,
            "target_release_gate": False,
        }
    if candidate_prediction.shape != truth.shape:
        raise ValueError("full SCPRC candidate shape does not match current fold")
    if not np.array_equal(candidate["label_starts"], starts):
        raise ValueError("full SCPRC candidate origins do not match current protocol")
    if not np.array_equal(candidate["site_ids"].astype(str), predictions["site_ids"]):
        raise ValueError("full SCPRC candidate site order does not match current protocol")
    if not np.array_equal(candidate["truth_raw"], truth):
        raise ValueError("full SCPRC candidate truth does not match current protocol")
    predictions["pred_scpcr_candidate_raw"] = candidate_prediction
    np.savez_compressed(out / "predictions.npz", **predictions)

    metrics = {
        arm: protocol._score(predictions[f"pred_{arm}_raw"], truth)
        for arm in ARMS
    }
    metrics["scpcr_candidate"] = protocol._score(candidate_prediction, truth)
    code_paths = (
        Path(__file__).resolve(),
        PROTOCOL_FILE,
        protocol.HERE,
        pcgrad.HERE,
        EXPERIMENTS / "phase_relations.py",
        ROOT / "scprc/baselines/phaseformer_adapter.py",
    )
    result = {
        "experiment": "SCPRC matched offline ablation matrix 2026-10-23",
        "status": "completed",
        "target_city": target,
        "source_cities": list(source_ids),
        "seed": seed,
        "protocol_id": PROTOCOL_ID,
        "protocol_file_id": PROTOCOL_ID,
        "analysis_protocol_id": "scpcr-matched-protocol-20261023",
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "retrospective": True,
        "blind_confirmation": False,
        "input": {
            "dataset": "CHARGED v3",
            "features": ["volume"],
            "target": "volume",
            "context_hours": 168,
            "horizon_hours": 24,
            "target_adaptation_steps": 0,
        },
        "data_split": {
            "source_window": "2023-04-01 through 2023-07-31",
            "evaluation_window": "2023-08-01 through 2023-08-31",
            "evaluation_origins": int(len(starts)),
        },
        "training_budget": {
            "source_minibatches_total": steps,
            "batch_size_windows": 4,
            "effective_source_windows": steps * 4,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "standardized volume MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
            "gradient_aggregation": "PCGrad across five source-city losses",
            "pcgrad_parameter_updates": (steps + len(source_ids) - 1) // len(source_ids),
            "torch_threads": torch_threads,
        },
        "arms": {
            "exact_pcgrad": "PhaseFormer Exact without relation adapter",
            "daily_relation_pcgrad": "phase-agnostic daily relation adapter; no source-certified attenuation",
            "phase_relation_pcgrad": "phase-conditioned relation adapter; no source-certified attenuation",
            "scpcr_candidate": "frozen current full SCPRC raw candidate reference; Online/fallback excluded",
        },
        "training": training,
        "metrics": metrics,
        "candidate_reference": {
            "result": str(candidate_dir / "result.json"),
            "predictions": str(candidate_dir / "predictions.npz"),
            "result_sha256": sha256(candidate_dir / "result.json"),
            "predictions_sha256": sha256(candidate_dir / "predictions.npz"),
            "protocol_id": candidate_result.get("current_run_provenance", {}).get("protocol_id"),
            "prediction_key": "pred_source_certified_internal_candidate_raw",
        },
        "checkpoints": {
            **{arm: str(path) for arm, path in checkpoints.items()},
            "scpcr_candidate": str(candidate_checkpoint),
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in code_paths if path.is_file()
        },
        "data_provenance": data_provenance,
        "wall_seconds": time.perf_counter() - started,
    }
    (out / "protocol.json").write_text(
        json.dumps({
            "protocol_id": PROTOCOL_ID,
            "canonical_protocol_file": str(PROTOCOL_FILE),
            "canonical_protocol_sha256": result["protocol_sha256"],
            "arms": result["arms"],
            "retrospective": True,
            "blind_confirmation": False,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "environment.json").write_text(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pid": os.getpid(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=protocol.CITY_CODES, required=True)
    parser.add_argument("--seed", choices=(13, 17, 29), type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--archive", type=Path, default=protocol.ARCHIVE)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    result = run_fold(
        args.target, args.seed, output_root=args.output_root, steps=args.steps,
        device=args.device, torch_threads=args.torch_threads,
        archive_path=args.archive,
    )
    print(json.dumps({
        "target": result["target_city"],
        "seed": result["seed"],
        "metrics": result["metrics"],
        "output": str(args.output_root / f"target_{args.target.lower()}_seed{args.seed}"),
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

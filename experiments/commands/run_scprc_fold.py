#!/usr/bin/env python3
"""Retrain the full SCPRC candidate for one fold under the frozen protocol.

The three control-arm predictions and checkpoints are copied from the already
completed matched-protocol fold.  Only the full candidate is retrained, so the
resulting matrix has one homogeneous, explicit full-candidate lineage across
all 18 city-seed folds without spending another run on the controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
SCRIPTS = ROOT
SOURCE_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261026"
OUTPUT_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiments.commands.run_matched_ablation_fold as matched  # noqa: E402


ARMS = (
    "exact_pcgrad",
    "daily_relation_pcgrad",
    "phase_relation_pcgrad",
)
SEEDS = (13, 17, 29)


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


def copy_control_artifacts(source_result: dict, source_dir: Path, output_dir: Path) -> tuple[dict, dict]:
    source_predictions_path = source_dir / "predictions.npz"
    with np.load(source_predictions_path, allow_pickle=False) as archive:
        predictions = {
            key: archive[key].copy()
            for key in ("label_starts", "truth_raw", "site_ids")
        }
        for arm in ARMS:
            key = f"pred_{arm}_raw"
            if key not in archive.files:
                raise KeyError(f"missing control prediction {key}: {source_predictions_path}")
            predictions[key] = archive[key].copy()

    training = {}
    checkpoints = {}
    for arm in ARMS:
        source_checkpoint = project_artifact_path(source_result["checkpoints"][arm])
        output_checkpoint = output_dir / f"{arm}.pt"
        shutil.copy2(source_checkpoint, output_checkpoint)
        checkpoints[arm] = output_checkpoint
        training[arm] = {
            **source_result["training"][arm],
            "reference_type": "copied_from_matched_control_fold",
            "source_result": str(source_dir / "result.json"),
            "source_result_sha256": sha256(source_dir / "result.json"),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256(source_checkpoint),
        }
    return predictions, {"training": training, "checkpoints": checkpoints}


def run_fold(
    target: str,
    seed: int,
    *,
    output_root: Path = OUTPUT_ROOT,
    steps: int = 2048,
    device: str = "cuda",
    torch_threads: int = 2,
    archive_path: Path = matched.protocol.ARCHIVE,
) -> dict:
    target = target.upper()
    if target not in matched.protocol.CITY_CODES:
        raise ValueError(f"invalid target {target}")
    if seed not in SEEDS:
        raise ValueError("the frozen matrix uses seeds 13, 17, and 29")
    if steps != 2048:
        raise ValueError("the frozen matrix requires exactly 2048 source minibatches")

    output_dir = output_root / f"target_{target.lower()}_seed{seed}"
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = SOURCE_ROOT / f"target_{target.lower()}_seed{seed}"
    source_result_path = source_dir / "result.json"
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    canonical_protocol_sha256 = matched.sha256(matched.PROTOCOL_FILE)
    if source_result.get("protocol_sha256") != canonical_protocol_sha256:
        raise ValueError(f"source fold has unexpected protocol hash: {source_result_path}")

    predictions, control_artifacts = copy_control_artifacts(source_result, source_dir, output_dir)
    cities, data_provenance = matched.protocol.load_cities(archive_path)
    source_ids = tuple(city for city in matched.protocol.CITY_CODES if city != target)
    mean, std = matched.protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = matched.protocol._validation_data(cities[target])
    if not np.array_equal(predictions["label_starts"], starts):
        raise ValueError("source control origins do not match the frozen protocol")
    if not np.array_equal(predictions["truth_raw"], truth):
        raise ValueError("source control truth does not match the frozen protocol")
    site_ids = np.asarray(cities[target]["site_ids"], dtype=str)
    if not np.array_equal(predictions["site_ids"].astype(str), site_ids):
        raise ValueError("source control site order does not match the frozen protocol")

    started = time.perf_counter()
    model, candidate_info = matched.train_offline_full_candidate(
        cities,
        source_ids,
        len(site_ids),
        mean,
        std,
        seed=seed,
        steps=steps,
        device=device,
        torch_threads=torch_threads,
    )
    candidate_prediction = matched.protocol._predict(
        model,
        target_values,
        starts,
        mean,
        std,
        device=device,
        batch_size=4,
    ).astype(np.float32)
    if candidate_prediction.shape != truth.shape or not np.isfinite(candidate_prediction).all():
        raise RuntimeError(f"unexpected candidate prediction shape: {candidate_prediction.shape}")
    candidate_checkpoint = output_dir / "scpcr_candidate.pt"
    torch.save(
        {
            "arm": "scpcr_candidate",
            "model_name": "scprc",
            "support_scale": 1.0,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        },
        candidate_checkpoint,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions["pred_scpcr_candidate_raw"] = candidate_prediction
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    metrics = {
        arm: matched.protocol._score(predictions[f"pred_{arm}_raw"], truth)
        for arm in (*ARMS, "scpcr_candidate")
    }
    candidate_training = {
        **candidate_info,
        "reference_type": "new_offline_no_support_projection_training",
        "validation_raw_volume": metrics["scpcr_candidate"],
    }
    training = {**control_artifacts["training"], "scpcr_candidate": candidate_training}
    checkpoints = {
        **control_artifacts["checkpoints"],
        "scpcr_candidate": candidate_checkpoint,
    }
    code_paths = (
        Path(__file__).resolve(),
        Path(matched.__file__).resolve(),
        matched.PROTOCOL_FILE,
        matched.protocol.HERE,
        matched.pcgrad.HERE,
        matched.EXPERIMENTS / "phase_relations.py",
        ROOT / "scprc/baselines/phaseformer_adapter.py",
    )
    result = {
        "experiment": "SCPRC homogeneous full-candidate retraining matrix 2026-10-27",
        "status": "completed",
        "target_city": target,
        "source_cities": list(source_ids),
        "seed": seed,
        "protocol_id": matched.PROTOCOL_ID,
        "protocol_file_id": matched.PROTOCOL_ID,
        "analysis_protocol_id": "scpcr-matched-protocol-20261023",
        "protocol_sha256": matched.sha256(matched.PROTOCOL_FILE),
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
            "exact_pcgrad": "matched-protocol control copied from the existing completed fold",
            "daily_relation_pcgrad": "matched-protocol control copied from the existing completed fold",
            "phase_relation_pcgrad": "matched-protocol control copied from the existing completed fold",
            "scpcr_candidate": "newly trained full SCPRC raw candidate; Online/fallback/support projection excluded",
        },
        "training": training,
        "metrics": metrics,
        "control_matrix_reference": {
            "result": str(source_result_path),
            "result_sha256": sha256(source_result_path),
            "predictions": str(source_dir / "predictions.npz"),
            "predictions_sha256": sha256(source_dir / "predictions.npz"),
        },
        "checkpoints": {arm: str(path) for arm, path in checkpoints.items()},
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in code_paths
            if path.is_file()
        },
        "data_provenance": data_provenance,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": matched.PROTOCOL_ID,
                "canonical_protocol_file": str(matched.PROTOCOL_FILE),
                "canonical_protocol_sha256": result["protocol_sha256"],
                "arms": result["arms"],
                "retrospective": True,
                "blind_confirmation": False,
                "full_candidate_lineage": "new_offline_no_support_projection_training",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=matched.protocol.CITY_CODES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--archive", type=Path, default=matched.protocol.ARCHIVE)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    result = run_fold(
        args.target,
        args.seed,
        output_root=args.output_root,
        steps=args.steps,
        device=args.device,
        torch_threads=args.torch_threads,
        archive_path=args.archive,
    )
    print(
        json.dumps(
            {
                "target": result["target_city"],
                "seed": result["seed"],
                "metrics": result["metrics"],
                "output": str(args.output_root / f"target_{args.target.lower()}_seed{args.seed}"),
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

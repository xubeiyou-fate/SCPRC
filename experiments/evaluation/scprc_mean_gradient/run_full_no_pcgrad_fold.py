#!/usr/bin/env python3
"""Train one canonical Full-no-PCGrad h=24 held-out-city fold.

This runner is deliberately isolated from the old experiments.  It reuses only
the canonical data loader, model constructor, source scaler and loss protocol;
it does not read an old checkpoint or an old prediction.  The Full SCPRC
relation/scale/peer/disagreement adapter is retained, while the source-city
gradients are combined by their ordinary arithmetic mean instead of PCGrad.

At the full budget, 2,048 source minibatches are consumed with batch size 4
(8,192 source windows).  Five source-city gradients are averaged per optimizer
update, giving 410 AdamW updates for every held-out city and seed.
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
OUTPUT_DEFAULT = HERE.parent
PROTOCOL_ID = "scpcr-matched-baseline-20261023"
METHOD_ID = "full_scpcr_no_pcgrad"
FULL_STEPS = 2048
SEEDS = (13, 17, 29)

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The current canonical source adapter imports this historical module only for
# three split constants.  Keep the compatibility shim local to R8 so the
# repository's old runner and experiments remain untouched.
_legacy_name = "calibration_protocol_constants"
if _legacy_name not in sys.modules:
    _legacy = types.ModuleType(_legacy_name)
    _legacy.CONTEXT = 168
    _legacy.HORIZON = 24
    _legacy.TRAIN_END = 2928
    sys.modules[_legacy_name] = _legacy

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.scprc_training_support as v21  # noqa: E402
import scprc.training.source_calibration_support as cert  # noqa: E402


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


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model(n_sites: int, threshold: float) -> torch.nn.Module:
    """Build the same Full SCPRC architecture used by canonical Full."""
    return v21._build_model(n_sites, float(threshold), support_scale=1.0).float()


def _ordinary_gradient_train(
    model: torch.nn.Module,
    cities: dict[str, dict[str, Any]],
    source_ids: tuple[str, ...],
    mean: float,
    std: float,
    *,
    seed: int,
    steps: int,
    device: str,
    torch_threads: int,
) -> dict[str, Any]:
    """Train with ordinary mean gradient aggregation and the canonical budget."""
    seed_everything(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=v21.LR, weight_decay=1e-4)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    starts = protocol._train_starts()
    rng = np.random.default_rng(seed)
    n_tasks = len(source_ids)
    base, remainder = divmod(steps, n_tasks)
    per_city_updates = {
        city: base + (index < remainder)
        for index, city in enumerate(source_ids)
    }
    rounds = max(per_city_updates.values())
    updates_by_city = {city: 0 for city in source_ids}
    started = time.perf_counter()
    model.train()

    for update in range(rounds):
        active = [city for city in source_ids if update < per_city_updates[city]]
        task_grads: list[tuple[torch.Tensor | None, ...]] = []
        for city_id in active:
            x, y = protocol._make_batch(
                cities[city_id]["values"][:v21.TRAIN_END],
                starts,
                mean,
                std,
                rng,
                v21.BATCH_SIZE,
            )
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = protocol._objective(prediction.float(), y)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite ordinary-gradient output/loss at update {update + 1}, {city_id}"
                )
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if not all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(
                    f"non-finite ordinary-gradient gradient at update {update + 1}, {city_id}"
                )
            task_grads.append(gradients)
            updates_by_city[city_id] += 1

        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            values = [
                gradients[parameter_index]
                for gradients in task_grads
                if gradients[parameter_index] is not None
            ]
            if values:
                parameter.grad = torch.stack(values, dim=0).mean(dim=0)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return {
        "model": "scprc",
        "gradient_aggregation": "ordinary arithmetic mean of source-city gradients",
        "pcgrad_enabled": False,
        "pcgrad_projection": False,
        "optimizer_steps": rounds,
        "configured_source_minibatches": steps,
        "source_windows": int(steps * v21.BATCH_SIZE),
        "seed": seed,
        "steps": steps,
        "batch_size_windows": v21.BATCH_SIZE,
        "torch_threads": int(torch_threads),
        "source_cities": list(source_ids),
        "source_optimizer_updates_by_city": updates_by_city,
        "optimizer": "AdamW",
        "learning_rate": v21.LR,
        "weight_decay": 1e-4,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "parameters": sum(parameter.numel() for parameter in parameters),
        "training_seconds": time.perf_counter() - started,
    }


def fold_paths(fold_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": fold_dir / "full_no_pcgrad.pt",
        "predictions": fold_dir / "predictions.npz",
        "result": fold_dir / "result.json",
        "protocol": fold_dir / "protocol.json",
        "progress": fold_dir / "progress.json",
    }


def fold_complete(fold_dir: Path, *, steps: int) -> bool:
    paths = fold_paths(fold_dir)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        expected_status = "completed" if steps == FULL_STEPS else "completed_smoke"
        if result.get("status") != expected_status or int(result.get("steps", -1)) != steps:
            return False
        if result.get("method_id") != METHOD_ID or result.get("pcgrad_enabled") is not False:
            return False
        with np.load(paths["predictions"], allow_pickle=False) as archive:
            required = {"label_starts", "truth_raw", "site_ids", "prediction_raw"}
            if not required.issubset(archive.files):
                return False
            if not all(np.isfinite(archive[key]).all() for key in required - {"site_ids"}):
                return False
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        return (
            checkpoint.get("method_id") == METHOD_ID
            and checkpoint.get("pcgrad_enabled") is False
            and isinstance(checkpoint.get("state_dict"), dict)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
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
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_root = output_root.resolve()
    fold_dir = output_root / f"target_{target.lower()}_seed{seed}"
    paths = fold_paths(fold_dir)
    if fold_complete(fold_dir, steps=steps):
        return json.loads(paths["result"].read_text(encoding="utf-8"))
    if paths["result"].exists() and not resume:
        raise FileExistsError(
            f"incomplete or incompatible fold exists at {fold_dir}; use --resume to rebuild it"
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    progress = {
        "method_id": METHOD_ID,
        "target_city": target,
        "seed": seed,
        "steps": steps,
        "source_windows": int(steps * v21.BATCH_SIZE),
        "pcgrad_enabled": False,
        "status": "running",
        "updated_at_unix": time.time(),
    }
    atomic_write_json(paths["progress"], progress)
    started = time.perf_counter()

    try:
        protocol_digest = sha256(PROTOCOL_FILE)
        cities, data_provenance = protocol.load_cities(archive_path)
        source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
        mean, std = protocol.source_scaler(cities, source_ids)
        starts, truth, target_values = protocol._validation_data(cities[target])
        site_ids = np.asarray(cities[target]["site_ids"], dtype=str)
        threshold = cert._estimate_threshold(
            cities,
            source_ids,
            mean,
            std,
            seed=seed,
            device=device,
            quantile=0.90,
        )
        progress.update({"status": "training", "source_relation_threshold": float(threshold)})
        atomic_write_json(paths["progress"], progress)

        model = build_model(len(site_ids), float(threshold))
        training = _ordinary_gradient_train(
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
        prediction = protocol._predict(
            model,
            target_values,
            starts,
            mean,
            std,
            device=device,
            batch_size=v21.BATCH_SIZE,
        ).astype(np.float32)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid prediction shape/value: {prediction.shape}")

        checkpoint_payload = {
            "method_id": METHOD_ID,
            "model_name": "scprc",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_digest,
            "pcgrad_enabled": False,
            "gradient_aggregation": "ordinary arithmetic mean of source-city gradients",
            "source_relation_threshold": float(threshold),
            "support_scale": 1.0,
            "target_support_projection": False,
            "target_release_gate": False,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        atomic_torch_save(paths["checkpoint"], checkpoint_payload)
        atomic_write_npz(
            paths["predictions"],
            label_starts=starts,
            truth_raw=truth,
            site_ids=site_ids,
            prediction_raw=prediction,
        )
        status = "completed" if steps == FULL_STEPS else "completed_smoke"
        result = {
            "experiment": "canonical h24 Full-no-PCGrad retraining",
            "method_id": METHOD_ID,
            "status": status,
            "smoke": steps != FULL_STEPS,
            "target_city": target,
            "seed": seed,
            "source_cities": list(source_ids),
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_digest,
            "protocol_file": str(PROTOCOL_FILE),
            "steps": steps,
            "source_windows": int(steps * v21.BATCH_SIZE),
            "context_hours": protocol.CONTEXT,
            "horizon_hours": protocol.HORIZON,
            "pcgrad_enabled": False,
            "gradient_aggregation": "ordinary arithmetic mean of source-city gradients",
            "relation_structure": "Full SCPRC scale + peer + source-disagreement + phase relation",
            "training": training,
            "metrics": protocol._score(prediction, truth),
            "checkpoint": str(paths["checkpoint"]),
            "predictions": str(paths["predictions"]),
            "checkpoint_sha256": sha256(paths["checkpoint"]),
            "predictions_sha256": sha256(paths["predictions"]),
            "independent_retraining": True,
            "old_artifacts_read": False,
            "data_provenance": data_provenance,
            "source_scaler": {"mean": mean, "std": std},
            "source_relation_threshold": float(threshold),
            "wall_seconds": time.perf_counter() - started,
            "code_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    HERE,
                    PROTOCOL_FILE,
                    protocol.HERE,
                    v21.HERE,
                    cert.HERE,
                    EXPERIMENTS / "phase_relations.py",
                    ROOT / "scprc/models/scprc.py",
                )
                if path.is_file()
            },
        }
        atomic_write_json(
            paths["protocol"],
            {
                "method_id": METHOD_ID,
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": protocol_digest,
                "source_minibatches": steps,
                "source_windows": int(steps * v21.BATCH_SIZE),
                "batch_size_windows": v21.BATCH_SIZE,
                "context_hours": protocol.CONTEXT,
                "horizon_hours": protocol.HORIZON,
                "source_city_training": True,
                "relation_structure": "Full SCPRC scale + peer + source-disagreement + phase relation",
                "gradient_aggregation": "ordinary arithmetic mean of source-city gradients",
                "pcgrad_enabled": False,
                "optimizer": "AdamW",
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
        atomic_write_json(fold_dir / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
        atomic_write_json(paths["result"], result)
        progress.update({"status": status, "updated_at_unix": time.time()})
        atomic_write_json(paths["progress"], progress)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as error:
        progress.update({
            "status": "failed",
            "error": repr(error),
            "updated_at_unix": time.time(),
        })
        atomic_write_json(paths["progress"], progress)
        raise


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

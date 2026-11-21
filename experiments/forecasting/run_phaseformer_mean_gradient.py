#!/usr/bin/env python3
"""Strict no-PCGrad PhaseFormer control for the canonical ablation matrix.

This control keeps the canonical PCGrad experiment's source-city exposure and
optimizer-update semantics: every update computes one gradient per source
city, averages those gradients without projection, then performs one AdamW
step. Only the PCGrad conflict projection is removed.
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
ROOT = next(parent for parent in HERE.parents
            if (parent / 'scprc/training').is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'
OUTPUT_DEFAULT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
PROTOCOL_ID = "scpcr-matched-baseline-20261023"
FULL_STEPS = 2048
SEEDS = (13, 17, 29)

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The source-certified module imports an archived calibration module only for
# constants. The canonical runner uses the same compatibility shim; keep this
# strict control independent from any archived training artifact.
_legacy_name = "calibration_protocol_constants"
if _legacy_name not in sys.modules:
    _legacy = types.ModuleType(_legacy_name)
    _legacy.CONTEXT = 168
    _legacy.HORIZON = 24
    _legacy.TRAIN_END = 2928
    sys.modules[_legacy_name] = _legacy

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.train_scprc as pcgrad  # noqa: E402


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_no_pcgrad(model: torch.nn.Module, cities: dict[str, dict[str, Any]],
                    source_ids: tuple[str, ...], mean: float, std: float,
                    *, seed: int, steps: int, device: str,
                    torch_threads: int = 2) -> dict[str, Any]:
    """Train with mean source gradients and no conflict projection."""
    seed_everything(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pcgrad.LR,
                                  weight_decay=1e-4)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    starts = protocol._train_starts()
    rng = np.random.default_rng(seed)
    base, remainder = divmod(steps, len(source_ids))
    updates_by_city = {
        city: base + (index < remainder)
        for index, city in enumerate(source_ids)
    }
    rounds = max(updates_by_city.values())
    seen_by_city = {city: 0 for city in source_ids}
    started = time.perf_counter()
    model.train()
    for update in range(rounds):
        active = [city for city in source_ids
                  if update < updates_by_city[city]]
        task_grads = []
        for city_id in active:
            x, y = protocol._make_batch(
                cities[city_id]["values"][:pcgrad.v21.TRAIN_END],
                starts, mean, std, rng, pcgrad.BATCH_SIZE,
            )
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = protocol._objective(prediction.float(), y)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite output/loss at update {update + 1}, {city_id}")
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=False, create_graph=False,
                allow_unused=True,
            )
            if not all(g is None or torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(
                    f"non-finite gradient at update {update + 1}, {city_id}")
            task_grads.append(grads)
            seen_by_city[city_id] += 1

        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            values = [grads[parameter_index] for grads in task_grads
                      if grads[parameter_index] is not None]
            if values:
                parameter.grad = torch.stack(values, dim=0).mean(dim=0)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return {
        "model": "phaseformer_exact",
        "arm": "phaseformer_no_pcgrad",
        "gradient_aggregation": "mean of five source-city gradients; PCGrad disabled",
        "pcgrad_enabled": False,
        "optimizer_steps": rounds,
        "configured_source_minibatches": steps,
        "seed": seed,
        "steps": steps,
        "batch_size_windows": pcgrad.BATCH_SIZE,
        "torch_threads": int(torch_threads),
        "source_cities": list(source_ids),
        "source_optimizer_updates_by_city": seen_by_city,
        "optimizer": "AdamW",
        "learning_rate": pcgrad.LR,
        "weight_decay": 1e-4,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "parameters": sum(p.numel() for p in parameters),
        "training_seconds": time.perf_counter() - started,
    }


def fold_complete(fold_dir: Path, steps: int) -> bool:
    result_path = fold_dir / "result.json"
    checkpoint = fold_dir / "phaseformer_no_pcgrad.pt"
    predictions = fold_dir / "predictions.npz"
    if not result_path.is_file() or not checkpoint.is_file() or not predictions.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or int(result.get("steps", -1)) != steps:
            return False
        with np.load(predictions, allow_pickle=False) as archive:
            required = {"label_starts", "truth_raw", "site_ids",
                        "pred_phaseformer_no_pcgrad_raw"}
            return required.issubset(archive.files) and np.isfinite(
                archive["pred_phaseformer_no_pcgrad_raw"]).all()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_fold(target: str, seed: int, *, output_root: Path = OUTPUT_DEFAULT,
             steps: int = FULL_STEPS, device: str = "cuda",
             torch_threads: int = 2, archive_path: Path = protocol.ARCHIVE,
             resume: bool = False) -> dict[str, Any]:
    target = target.upper()
    if target not in protocol.CITY_CODES:
        raise ValueError(f"invalid target {target}")
    if seed not in SEEDS:
        raise ValueError(f"invalid seed {seed}; expected {SEEDS}")
    if steps != FULL_STEPS:
        raise ValueError("strict matrix requires exactly 2048 source minibatches")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_root = output_root.resolve()
    fold_dir = output_root / f"target_{target.lower()}_seed{seed}"
    result_path = fold_dir / "result.json"
    if fold_complete(fold_dir, steps):
        return json.loads(result_path.read_text(encoding="utf-8"))
    if result_path.exists() and not resume:
        raise FileExistsError(f"incomplete fold exists at {fold_dir}; use --resume")
    fold_dir.mkdir(parents=True, exist_ok=True)

    cities, data_provenance = protocol.load_cities(archive_path)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    site_ids = np.asarray(cities[target]["site_ids"], dtype=str)
    model = protocol._build_model("phaseformer_exact", len(site_ids)).float()
    info = train_no_pcgrad(
        model, cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads,
    )
    prediction = protocol._predict(
        model, target_values, starts, mean, std, device=device, batch_size=4,
    ).astype(np.float32)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise RuntimeError(f"unexpected prediction shape {prediction.shape}")

    checkpoint_path = fold_dir / "phaseformer_no_pcgrad.pt"
    atomic_torch(checkpoint_path, {
        "arm": "phaseformer_no_pcgrad",
        "model_name": "phaseformer_exact",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "pcgrad_enabled": False,
        "gradient_aggregation": "mean source-city gradients without projection",
        "state_dict": {key: value.detach().cpu()
                        for key, value in model.state_dict().items()},
    })
    prediction_path = fold_dir / "predictions.npz"
    atomic_npz(
        prediction_path,
        label_starts=starts,
        truth_raw=truth,
        site_ids=site_ids,
        prediction_raw=prediction,
        pred_phaseformer_no_pcgrad_raw=prediction,
    )
    result = {
        "experiment": "strict ablation: Original PhaseFormer without PCGrad",
        "status": "completed",
        "target_city": target,
        "seed": seed,
        "source_cities": list(source_ids),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "protocol_file": str(PROTOCOL_FILE),
        "context_hours": protocol.CONTEXT,
        "horizon_hours": protocol.HORIZON,
        "steps": steps,
        "arm": "phaseformer_no_pcgrad",
        "model_name": "phaseformer_exact",
        "pcgrad_enabled": False,
        "independent_retraining": True,
        "old_artifacts_read": False,
        "retrospective": True,
        "blind_confirmation": False,
        "training": {"phaseformer_no_pcgrad": info},
        "metrics": {"phaseformer_no_pcgrad": protocol._score(prediction, truth)},
        "checkpoint": str(checkpoint_path),
        "predictions": str(prediction_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions_sha256": sha256(prediction_path),
        "data_provenance": data_provenance,
        "source_scaler": {"mean": mean, "std": std},
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (HERE, PROTOCOL_FILE, protocol.HERE, pcgrad.HERE,
                         pcgrad.v21.HERE)
            if path.is_file()
        },
    }
    atomic_json(fold_dir / "protocol.json", {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": result["protocol_sha256"],
        "context_hours": protocol.CONTEXT,
        "horizon_hours": protocol.HORIZON,
        "source_minibatches": steps,
        "optimizer_steps": info["optimizer_steps"],
        "gradient_aggregation": info["gradient_aggregation"],
        "pcgrad_enabled": False,
    })
    atomic_json(fold_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_requested": device,
        "torch_threads": torch_threads,
        "pid": os.getpid(),
    })
    atomic_json(fold_dir / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    atomic_json(result_path, result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=protocol.CITY_CODES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--archive", type=Path, default=protocol.ARCHIVE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_fold(args.target, args.seed, output_root=args.output_root,
                      steps=args.steps, device=args.device,
                      torch_threads=args.torch_threads,
                      archive_path=args.archive, resume=args.resume)
    print(json.dumps({"status": result["status"], "target": result["target_city"],
                      "seed": result["seed"], "metrics": result["metrics"],
                      "output": str(project_artifact_path(result["predictions"]).parent)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

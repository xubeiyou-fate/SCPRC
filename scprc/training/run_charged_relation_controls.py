#!/usr/bin/env python3
"""Run the two frozen-protocol relation controls on CHARGED v3.

The N-BEATS-style behavior-graph model is an independent local implementation,
not a scprc of the cited Energy paper. The attention model swaps only
SCPRC's input relation adapter for a parameter-matched generic set-attention
adapter. Both use source-only PCGrad and the retrospective August audit split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = HERE.parent
for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scprc.training.training_protocol as protocol  # noqa: E402
from scprc.training.nbeats_graph import (  # noqa: E402
    SharedStationNBEATSBehaviorGraphResidual,
)
from scprc.models.phasewise_attention_set_adapter import (  # noqa: E402
    PhaseWiseAttentionPhaseFormer,
)

MODELS = (
    "NBEATS_behavior_graph",
    "PhaseWiseAttention_relation",
)
SEEDS = (13, 17, 29)
DEFAULT_OUTPUTS = {
    "NBEATS_behavior_graph": EXPERIMENTS / "charged_nbeats_behavior_graph_pcgrad_loco_20261023",
    "PhaseWiseAttention_relation": EXPERIMENTS / "charged_phasewise_attention_pcgrad_loco_20261023",
}
FROZEN_PROTOCOL_PATH = EXPERIMENTS / 'scprc_matched_protocol.json'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model(model_name: str, site_count: int) -> torch.nn.Module:
    if model_name == "NBEATS_behavior_graph":
        return SharedStationNBEATSBehaviorGraphResidual(
            hidden=128, blocks=4, graph_rank=16, temperature=0.15,
            residual_bound=0.20,
        )
    if model_name == "PhaseWiseAttention_relation":
        return PhaseWiseAttentionPhaseFormer(
            channels=site_count, seq_len=protocol.CONTEXT,
            pred_len=protocol.HORIZON, period_len=24, latent_dim=32,
            phase_layers=2, routers=1, heads=8, dropout=0.1,
            pos_embed=True, residual_bound=0.2,
        )
    raise ValueError(f"unknown model: {model_name}")


def _global_dot(left: tuple[torch.Tensor | None, ...],
                right: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    terms = [(a * b).sum() for a, b in zip(left, right)
             if a is not None and b is not None]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum()


def _project_conflicts(task_grads: list[tuple[torch.Tensor | None, ...]],
                       order: np.ndarray) -> tuple[list[tuple[torch.Tensor | None, ...]], int, int]:
    original = task_grads
    projected: list[list[torch.Tensor | None]] = [
        [grad.clone() if grad is not None else None for grad in gradients]
        for gradients in original
    ]
    conflicts = 0
    pairs = 0
    for index in order:
        for other_index in order:
            if index == other_index:
                continue
            dot = _global_dot(tuple(projected[index]), original[other_index])
            norm = _global_dot(original[other_index], original[other_index]).clamp_min(1e-20)
            pairs += 1
            if dot.item() < 0.0:
                conflicts += 1
                ratio = dot / norm
                for parameter_index, other in enumerate(original[other_index]):
                    current = projected[index][parameter_index]
                    if current is not None and other is not None:
                        current.add_(other, alpha=-float(ratio.item()))
    return [tuple(gradients) for gradients in projected], conflicts, pairs


def _train_pcgrad(model: torch.nn.Module, cities: dict[str, dict[str, Any]],
                  source_ids: tuple[str, ...], mean: float, std: float, *,
                  seed: int, steps: int, device: str,
                  torch_threads: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, torch_threads))
    model = model.float().to(device)
    parameters = tuple(parameter for parameter in model.parameters()
                       if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    starts = protocol._train_starts()
    batch_rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed ^ 0x5C2C)
    base, remainder = divmod(steps, len(source_ids))
    per_city = {city: base + (index < remainder)
                for index, city in enumerate(source_ids)}
    rounds = max(per_city.values())
    updates_by_city = {city: 0 for city in source_ids}
    conflict_pairs = 0
    pair_count = 0
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    model.train()
    for update in range(rounds):
        active_cities = [city for city in source_ids if update < per_city[city]]
        task_grads: list[tuple[torch.Tensor | None, ...]] = []
        for city in active_cities:
            x, y = protocol._make_batch(
                cities[city]["values"][:protocol.TRAIN_END], starts,
                mean, std, batch_rng, protocol.BATCH_SIZE,
            )
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = protocol._objective(prediction.float(), y)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at update {update + 1} for {city}")
            gradients = torch.autograd.grad(
                loss, parameters, allow_unused=True,
            )
            if not all(grad is None or torch.isfinite(grad).all()
                       for grad in gradients):
                raise FloatingPointError(
                    f"non-finite gradient at update {update + 1} for {city}")
            task_grads.append(gradients)
            updates_by_city[city] += 1
        if len(task_grads) > 1:
            projected, conflicts, pairs = _project_conflicts(
                task_grads, order_rng.permutation(len(task_grads)))
            conflict_pairs += conflicts
            pair_count += pairs
        else:
            projected = task_grads
        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            gradients = [task[parameter_index] for task in projected
                         if task[parameter_index] is not None]
            if gradients:
                parameter.grad = torch.stack(gradients).mean(dim=0)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if (update + 1) % 50 == 0 or update + 1 == rounds:
            print(json.dumps({"event": "training_progress", "update": update + 1,
                              "rounds": rounds, "elapsed_s": round(
                                  time.perf_counter() - started, 2)},
                             separators=(",", ":")), flush=True)
    training_seconds = time.perf_counter() - started
    peak_memory = (torch.cuda.max_memory_allocated(torch.device(device))
                   if torch.cuda.is_available() and torch.device(device).type == "cuda"
                   else 0)
    return {
        "gradient_aggregation": "seeded source-city PCGrad pairwise projection",
        "gradient_projection_order": "seeded shuffle per update",
        "configured_source_minibatches": steps,
        "effective_source_windows": steps * protocol.BATCH_SIZE,
        "optimizer_steps": rounds,
        "source_optimizer_updates_by_city": updates_by_city,
        "source_cities": list(source_ids),
        "batch_size_windows_per_source_minibatch": protocol.BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "loss": "standardized volume MSE + 0.15 multiresolution consistency (2,3,4,6,12)",
        "source_gradient_conflict_pairs": conflict_pairs,
        "source_gradient_pair_count": pair_count,
        "source_gradient_conflict_fraction": conflict_pairs / pair_count if pair_count else 0.0,
        "parameters": sum(parameter.numel() for parameter in parameters),
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": int(peak_memory),
        "precision": "float32",
    }


def _predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
             mean: float, std: float, *, device: str) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), protocol.BATCH_SIZE):
            batch_starts = starts[offset:offset + protocol.BATCH_SIZE]
            x = np.stack([values[start - protocol.CONTEXT:start]
                          for start in batch_starts]).astype(np.float32)
            x = torch.from_numpy((x - mean) / std).to(device)
            prediction = model(x)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("non-finite held-out forecast")
            chunks.append(prediction.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32) * std + mean


def _metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error)))}


def run_fold(model_name: str, target: str, seed: int, *,
             cities: dict[str, dict[str, Any]], data_provenance: dict[str, Any],
             output_root: Path, steps: int, device: str,
             torch_threads: int, overwrite: bool = False) -> dict[str, Any]:
    out = output_root / f"target_{target.lower()}_seed{seed}"
    result_path = out / "result.json"
    prediction_path = out / "predictions.npz"
    checkpoint_path = out / "trained_checkpoint.pt"
    if not overwrite and all(path.is_file() for path in
                             (result_path, prediction_path, checkpoint_path)):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            print(json.dumps({"event": "fold_skipped_complete", "model": model_name,
                              "target": target, "seed": seed}, separators=(",", ":")),
                  flush=True)
            return cached
    out.mkdir(parents=True, exist_ok=True)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    if len(starts) != 721:
        raise ValueError(f"expected 721 August origins, got {len(starts)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _build_model(model_name, len(cities[target]["site_ids"]))
    train_info = _train_pcgrad(
        model, cities, source_ids, mean, std, seed=seed, steps=steps,
        device=device, torch_threads=torch_threads,
    )
    if model_name == "PhaseWiseAttention_relation":
        train_info["adapter_parameters"] = sum(
            parameter.numel() for parameter in model.relation.parameters()
        )
    model.eval()
    prediction = _predict(model, target_values, starts, mean, std, device=device)
    if prediction.shape != truth.shape:
        raise ValueError(f"forecast shape {prediction.shape} != truth shape {truth.shape}")
    scores = _metric(prediction, truth)

    checkpoint_tmp = out / "trained_checkpoint.pt.tmp"
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "target_city": target,
        "seed": seed,
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "model_config": ({"hidden": 128, "blocks": 4, "graph_rank": 16,
                          "temperature": 0.15, "residual_bound": 0.20}
                         if model_name == "NBEATS_behavior_graph" else
                         {"backbone": "PhaseFormer exact", "adapter": "one-head phase-wise station self-attention",
                          "adapter_parameters": sum(p.numel() for p in model.relation.parameters()),
                          "residual_bound": 0.2}),
        "protocol_id": "scpcr-matched-baseline-20261023",
    }, checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint_path)
    prediction_key = (
        "prediction_nbeats_behavior_graph_raw"
        if model_name == "NBEATS_behavior_graph"
        else "prediction_phasewise_attention_raw"
    )
    prediction_tmp = out / "predictions.npz.tmp"
    with prediction_tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            label_starts=starts,
            label_timestamps=np.asarray([
                str(cities[target]["times"][int(start)]) for start in starts
            ]),
            site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
            truth_raw=truth.astype(np.float32),
            **{prediction_key: prediction.astype(np.float32)},
        )
    os.replace(prediction_tmp, prediction_path)

    code_paths = [
        HERE,
        FROZEN_PROTOCOL_PATH,
        EXPERIMENTS / "training_protocol.py",
        EXPERIMENTS / "train_scprc.py",
        EXPERIMENTS / "nbeats_graph.py",
        ROOT / "scprc/models/phasewise_attention_set_adapter.py",
        ROOT / "scprc/baselines/phaseformer_adapter.py",
        ROOT / "scprc/data_adapters/charged/city_adapter.py",
    ]
    result = {
        "status": "completed",
        "experiment": "CHARGED six-city source-only relation control; retrospective August audit",
        "model": model_name,
        "model_scope": (
            "Independent local N-BEATS-style + history-conditioned dynamic behavior-graph residual; not a scprc of the cited paper"
            if model_name == "NBEATS_behavior_graph" else
            "PhaseFormer exact backbone with parameter-matched history-conditioned phase-wise attention set adapter"
        ),
        "original_paper_reproduction": False,
        "target_city": target,
        "source_cities": list(source_ids),
        "seed": seed,
        "metrics": scores,
        "prediction_key": prediction_key,
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "training": train_info,
        "artifacts": {
            "prediction_archive": str(prediction_path),
            "prediction_archive_sha256": _sha256(prediction_path),
            "trained_checkpoint": str(checkpoint_path),
            "trained_checkpoint_sha256": _sha256(checkpoint_path),
        },
        "protocol": {
            "protocol_id": "scpcr-matched-baseline-20261023",
            "input": "volume-only",
            "context_hours": 168,
            "horizon_hours": 24,
            "source_train_interval": "2023-04-01 00:00 through 2023-07-31 23:00",
            "source_train_end_exclusive_index": protocol.TRAIN_END,
            "evaluation_interval": "2023-08-01 00:00 through 2023-08-31 23:00",
            "evaluation_origins": int(len(starts)),
            "target_adaptation_steps": 0,
            "target_city_labels_used_for_optimization": False,
            "target_city_statistics_used_for_scaling": False,
            "source_scaler": "equal-city-weight global mean/std over the five source cities",
            "source_minibatches": steps,
            "effective_source_windows": steps * protocol.BATCH_SIZE,
            "pcgrad_optimizer_updates": train_info["optimizer_steps"],
            "september_values_materialized": False,
            "august_role": "retrospective development/audit only; not blind confirmation",
            "data_provenance": data_provenance,
            "frozen_protocol_sha256": _sha256(FROZEN_PROTOCOL_PATH),
            "code_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                            for path in code_paths if path.is_file()},
        },
    }
    result_tmp = out / "result.json.tmp"
    result_tmp.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(result_tmp, result_path)
    print(json.dumps({"event": "fold_completed", "model": model_name,
                      "target": target, "seed": seed, "metrics": scores,
                      "training_seconds": train_info["training_seconds"],
                      "checkpoint": str(checkpoint_path)},
                     separators=(",", ":")), flush=True)
    del model
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS + ("all",), default="all")
    parser.add_argument("--target", choices=protocol.CITY_CODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.target is None) != (args.seed is None):
        parser.error("--target and --seed must be supplied together")
    if args.steps != 2048 and (args.target is None or args.seed is None):
        parser.error("non-frozen smoke budgets require one explicit --target and --seed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    selected_models = MODELS if args.model == "all" else (args.model,)
    targets = (args.target,) if args.target else protocol.CITY_CODES
    seeds = (args.seed,) if args.seed else SEEDS
    print(json.dumps({"event": "loading_frozen_dataset", "models": selected_models,
                      "targets": targets, "seeds": seeds, "device": args.device},
                     separators=(",", ":")), flush=True)
    cities, provenance = protocol.load_cities()
    failed = []
    for model_name in selected_models:
        for target in targets:
            for seed in seeds:
                try:
                    run_fold(
                        model_name, target, seed, cities=cities,
                        data_provenance=provenance,
                        output_root=DEFAULT_OUTPUTS[model_name], steps=args.steps,
                        device=args.device, torch_threads=args.torch_threads,
                        overwrite=args.overwrite,
                    )
                except Exception as error:
                    failed.append({"model": model_name, "target": target,
                                   "seed": seed, "error": repr(error)})
                    failure_dir = DEFAULT_OUTPUTS[model_name] / f"target_{target.lower()}_seed{seed}"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    (failure_dir / "failure.json").write_text(
                        json.dumps(failed[-1], indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"event": "fold_failed", **failed[-1]},
                                     separators=(",", ":")), flush=True)
    print(json.dumps({"event": "matrix_finished", "failed_folds": failed,
                      "completed_expected_folds": len(selected_models) * len(targets) * len(seeds) - len(failed)},
                     separators=(",", ":")), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

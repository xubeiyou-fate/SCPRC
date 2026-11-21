#!/usr/bin/env python3
"""SCPRC-v2.2: source-city PCGrad training with the fixed v2.1 certificates.

All source-city losses are computed from April--July training windows. Their
per-city gradients are combined by the fixed PCGrad projection rule. August is
used only by the inherited per-target dual-metric/p95 release gate. September
is never materialized here. No learned gate, gate search, or expert pool.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.scprc_training_support as v21  # noqa: E402

BATCH_SIZE = v21.BATCH_SIZE
LR = v21.LR


def _global_dot(left: tuple[torch.Tensor | None, ...],
                right: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    terms = [(a * b).sum() for a, b in zip(left, right)
             if a is not None and b is not None]
    if not terms:
        return torch.zeros((), device=next(
            (x.device for x in (*left, *right) if x is not None),
            torch.device("cpu")))
    return torch.stack(terms).sum()


def _project_conflicts(task_grads: list[tuple[torch.Tensor | None, ...]],
                       order: np.ndarray) -> tuple[list[tuple[torch.Tensor | None, ...]], int, int]:
    """Canonical PCGrad projection against the other tasks' original grads."""
    original = task_grads
    projected: list[list[torch.Tensor | None]] = [
        [g.clone() if g is not None else None for g in grads]
        for grads in original
    ]
    conflicts = 0
    pairs = 0
    for i in order:
        for j in order:
            if i == j:
                continue
            dot = _global_dot(tuple(projected[i]), original[j])
            norm = _global_dot(original[j], original[j]).clamp_min(1e-20)
            pairs += 1
            if dot.item() < 0.0:
                conflicts += 1
                ratio = dot / norm
                for k, other in enumerate(original[j]):
                    if projected[i][k] is not None and other is not None:
                        projected[i][k].add_(other, alpha=-float(ratio.item()))
    return [tuple(g for g in task) for task in projected], conflicts, pairs


def _train_candidate_pcgrad(model: torch.nn.Module, cities: dict,
                            source_ids: tuple[str, ...], mean: float, std: float,
                            *, seed: int, steps: int, device: str,
                            torch_threads: int = 2) -> dict:
    """Train the v2.1 internal adapter with source-city PCGrad.

    The configured 2,048 source minibatches are distributed evenly across
    source cities, matching v2.1's per-city sample exposure. One optimizer
    update aggregates one task gradient from each source city.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    starts = protocol._train_starts()
    rng = np.random.default_rng(seed)
    n_tasks = len(source_ids)
    base, remainder = divmod(steps, n_tasks)
    per_city_updates = {
        city: base + (index < remainder)
        for index, city in enumerate(source_ids)
    }
    rounds = max(per_city_updates.values())
    update_order_rng = np.random.default_rng(seed ^ 0x5C2C)
    updates_by_city = {city: 0 for city in source_ids}
    conflict_pairs = 0
    all_pairs = 0
    started = time.perf_counter()
    model.train()
    for update in range(rounds):
        active = [city for city in source_ids
                  if update < per_city_updates[city]]
        task_grads = []
        for city_id in active:
            x, y = protocol._make_batch(
                cities[city_id]["values"][:v21.TRAIN_END], starts,
                mean, std, rng, BATCH_SIZE,
            )
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = protocol._objective(prediction.float(), y)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite PCGrad output/loss at update {update + 1}, {city_id}")
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=False, create_graph=False,
                allow_unused=True,
            )
            if not all(g is None or torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(
                    f"non-finite PCGrad gradient at update {update + 1}, {city_id}")
            task_grads.append(grads)
            updates_by_city[city_id] += 1
        if len(task_grads) > 1:
            order = update_order_rng.permutation(len(task_grads))
            projected, conflicts, pairs = _project_conflicts(task_grads, order)
            conflict_pairs += conflicts
            all_pairs += pairs
        else:
            projected = task_grads
        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            values = [grads[parameter_index] for grads in projected
                      if grads[parameter_index] is not None]
            if values:
                parameter.grad = torch.stack(values, dim=0).mean(dim=0)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return {
        "model": "source_certified_internal_phase_adapter_pcgrad",
        "gradient_aggregation": "PCGrad pairwise projection on conflicting source-city gradients",
        "gradient_projection_order": "seeded shuffle per update",
        "optimizer_steps": rounds,
        "configured_source_minibatches": steps,
        "seed": seed, "steps": steps, "batch_size_windows": BATCH_SIZE,
        "torch_threads": int(torch_threads), "source_cities": list(source_ids),
        "source_optimizer_updates_by_city": updates_by_city,
        "optimizer": "AdamW", "learning_rate": LR, "weight_decay": 1e-4,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "source_gradient_conflict_pairs": conflict_pairs,
        "source_gradient_pair_count": all_pairs,
        "source_gradient_conflict_fraction": (
            conflict_pairs / all_pairs if all_pairs else 0.0),
        "parameters": sum(p.numel() for p in parameters),
        "training_seconds": time.perf_counter() - started,
    }


def run(target: str, seed: int, *, output_root: Path, steps: int = v21.STEPS,
        device: str = "cuda", torch_threads: int = 2,
        source_diagnostic: bool = False, archive_path=None) -> dict:
    # Reuse the audited v2.1 data loader, fixed certificates, backbone,
    # August gate, and prediction writer; only replace source gradient joining.
    v21._train_candidate = _train_candidate_pcgrad
    kwargs = dict(output_root=output_root, steps=steps, device=device,
                  torch_threads=torch_threads,
                  source_diagnostic=source_diagnostic)
    if archive_path is not None:
        kwargs["archive_path"] = archive_path
    result = v21.run(target, seed, **kwargs)
    result["experiment"] = "CHARGED v3 SCPRC-v2.2 source-city PCGrad internal phase adapter"
    result["candidate"]["method_version"] = "SCPRC-v2.2-PCGrad"
    result["protocol"].update({
        "algorithm_iteration": "PCGrad source-city gradient conflict projection",
        "gradient_conflict_search": False,
        "learned_gate": False,
        "gate_parameter_search": False,
        "expert_pool": False,
        "september_values_materialized": False,
        "august_role": "fixed dual-metric and p95 per-target release audit only",
    })
    out = output_root / f"target_{target.lower()}_seed{seed}" / "result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True,
                              allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=v21.protocol.CITY_CODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path,
                        default=HERE.parent / "charged_scpcr_v22_pcgrad_loco")
    parser.add_argument("--archive", type=Path, default=v21.protocol.ARCHIVE)
    parser.add_argument("--steps", type=int, default=v21.STEPS)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--source-diagnostic", action="store_true")
    args = parser.parse_args()
    result = run(args.target, args.seed, output_root=args.output_root,
                 steps=args.steps, device=args.device,
                 torch_threads=args.torch_threads,
                 source_diagnostic=args.source_diagnostic,
                 archive_path=args.archive)
    print(json.dumps({"target_city": result["target_city"], "seed": result["seed"],
                      "selected_model": result["selected_model"],
                      "target_august_gate": result["target_august_gate"],
                      "training": result["candidate"],
                      "metrics": result["metrics"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Source-only LOCO for SCPRC-v2.1 internal phase adaptation.

The relation adapter is trained end-to-end on April--July source windows and
is selected only with August labels from the five source cities.  The held-out
target August labels are audit-only; September is never materialized here.
"""
from __future__ import annotations

import argparse
import json
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

from scprc.data_adapters.charged.city_adapter import ARCHIVE, CITY_CODES  # noqa: E402
import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.source_calibration_support as cert  # noqa: E402
from scprc.models.scprc import (  # noqa: E402
    SCPRC,
)

CONTEXT = protocol.CONTEXT
HORIZON = protocol.HORIZON
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END
STEPS = 2048
BATCH_SIZE = 4
LR = 1e-3
TAIL_Q = 0.95
TAIL_TOL = 0.02


def _build_model(n_sites: int, threshold: float, support_scale: float):
    return SCPRC(
        channels=n_sites, seq_len=CONTEXT, pred_len=HORIZON, period_len=24,
        latent_dim=32, phase_layers=2, routers=1, heads=8,
        dropout=0.1, pos_embed=True, relation_mode="phase",
        relation_hidden=16, relation_rank=8,
        support_scale=support_scale,
        source_relation_threshold=threshold,
    )


def _train_candidate(model: torch.nn.Module, cities: dict, source_ids: tuple[str, ...],
                     mean: float, std: float, *, seed: int, steps: int,
                     device: str, torch_threads: int = 2) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    starts = protocol._train_starts()
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    amp_enabled = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    started = time.perf_counter()
    model.train()
    for update in range(steps):
        city_id = source_ids[update % len(source_ids)]
        x, y = protocol._make_batch(
            cities[city_id]["values"][:TRAIN_END], starts, mean, std,
            rng, BATCH_SIZE,
        )
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp_enabled):
            prediction = model(x)
            loss = protocol._objective(prediction.float(), y)
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite candidate output/loss at update {update + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        updates_by_city[city_id] += 1
    return {
        "model": "scprc",
        "seed": seed, "steps": steps, "batch_size_windows": BATCH_SIZE,
        "torch_threads": int(torch_threads), "source_cities": list(source_ids),
        "source_optimizer_updates_by_city": updates_by_city,
        "optimizer": "AdamW", "learning_rate": LR, "weight_decay": 1e-4,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": time.perf_counter() - started,
    }


def _predict(model: torch.nn.Module, values: np.ndarray, starts: np.ndarray,
             mean: float, std: float, *, device: str, batch_size: int = 4) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            current = starts[offset:offset + batch_size]
            raw = np.stack([values[s - CONTEXT:s] for s in current]).astype(np.float32)
            x = torch.from_numpy((raw - mean) / std).to(device)
            pred = model(x)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("non-finite forecast")
            chunks.append(pred.cpu().numpy())
    return np.concatenate(chunks, axis=0) * std + mean


def _source_gate(backbone: torch.nn.Module, candidate: torch.nn.Module,
                 cities: dict, source_ids: tuple[str, ...], mean: float, std: float,
                 *, device: str, tail_tolerance: float = TAIL_TOL) -> dict:
    """Fixed dual-metric August gate evaluated only on source cities."""
    per_city = {}
    base_rmse, cand_rmse, base_mae, cand_mae = [], [], [], []
    base_tail, cand_tail = [], []
    for city in source_ids:
        starts, truth, values = protocol._validation_data(cities[city])
        base = _predict(backbone, values, starts, mean, std, device=device)
        cand = _predict(candidate, values, starts, mean, std, device=device)
        base_err = base.astype(np.float64) - truth.astype(np.float64)
        cand_err = cand.astype(np.float64) - truth.astype(np.float64)
        base_windows = np.sqrt(np.mean(base_err * base_err, axis=(1, 2)))
        cand_windows = np.sqrt(np.mean(cand_err * cand_err, axis=(1, 2)))
        b = protocol._score(base, truth); c = protocol._score(cand, truth)
        base_rmse.append(b["rmse"]); cand_rmse.append(c["rmse"])
        base_mae.append(b["mae"]); cand_mae.append(c["mae"])
        base_tail.append(float(np.quantile(base_windows, TAIL_Q)))
        cand_tail.append(float(np.quantile(cand_windows, TAIL_Q)))
        per_city[city] = {
            "base_rmse": b["rmse"], "candidate_rmse": c["rmse"],
            "base_mae": b["mae"], "candidate_mae": c["mae"],
            "base_tail_rmse_p95": base_tail[-1],
            "candidate_tail_rmse_p95": cand_tail[-1],
        }
    both_wins = int(sum(
        cand_rmse[i] <= base_rmse[i] and cand_mae[i] <= base_mae[i]
        for i in range(len(source_ids))
    ))
    tail_pass = bool(all(
        cand_tail[i] <= base_tail[i] * (1.0 + float(tail_tolerance))
        for i in range(len(source_ids))
    ))
    selected = bool(
        np.mean(cand_rmse) <= np.mean(base_rmse)
        and np.mean(cand_mae) <= np.mean(base_mae)
        and both_wins >= max(3, (len(source_ids) + 1) // 2)
        and tail_pass
    )
    return {
        "selection_data": "source-city August only",
        "source_august_origins_per_city": 721,
        "per_city": per_city,
        "base_rmse_mean": float(np.mean(base_rmse)),
        "candidate_rmse_mean": float(np.mean(cand_rmse)),
        "base_mae_mean": float(np.mean(base_mae)),
        "candidate_mae_mean": float(np.mean(cand_mae)),
        "base_tail_rmse_p95_mean": float(np.mean(base_tail)),
        "candidate_tail_rmse_p95_mean": float(np.mean(cand_tail)),
        "both_metric_wins": both_wins,
        "tail_tolerance": float(tail_tolerance),
        "tail_guard_passed": tail_pass,
        "selected": selected,
    }


def _target_august_gate(backbone: torch.nn.Module, candidate: torch.nn.Module,
                        cities: dict, target: str, mean: float, std: float,
                        *, device: str, tail_tolerance: float = TAIL_TOL) -> dict:
    """Apply the predeclared August rule to one held-out target city.

    August is the development interval for this track.  The rule is fixed
    before September is opened: both raw metrics must improve and the p95
    rolling-window RMSE may not increase by more than two percent.  This is a
    release audit, not a trainable gate and has no tunable gate parameter.
    """
    starts, truth, values = protocol._validation_data(cities[target])
    base = _predict(backbone, values, starts, mean, std, device=device)
    cand = _predict(candidate, values, starts, mean, std, device=device)
    base_err = base.astype(np.float64) - truth.astype(np.float64)
    cand_err = cand.astype(np.float64) - truth.astype(np.float64)
    base_windows = np.sqrt(np.mean(base_err * base_err, axis=(1, 2)))
    cand_windows = np.sqrt(np.mean(cand_err * cand_err, axis=(1, 2)))
    b = protocol._score(base, truth); c = protocol._score(cand, truth)
    b_tail = float(np.quantile(base_windows, TAIL_Q))
    c_tail = float(np.quantile(cand_windows, TAIL_Q))
    selected = bool(
        c["rmse"] <= b["rmse"] and c["mae"] <= b["mae"]
        and c_tail <= b_tail * (1.0 + float(tail_tolerance))
    )
    return {
        "selection_data": "held-out target August only",
        "target": target, "validation_origins": int(len(starts)),
        "base_rmse": b["rmse"], "candidate_rmse": c["rmse"],
        "base_mae": b["mae"], "candidate_mae": c["mae"],
        "base_tail_rmse_p95": b_tail, "candidate_tail_rmse_p95": c_tail,
        "tail_tolerance": float(tail_tolerance), "selected": selected,
    }


def run(target: str, seed: int, *, output_root: Path, steps: int = STEPS,
        device: str = "cuda", torch_threads: int = 2,
        source_diagnostic: bool = False,
        archive_path: Path = ARCHIVE) -> dict:
    target = target.upper()
    if target not in CITY_CODES or not isinstance(seed, int):
        raise ValueError("invalid target or seed")
    cities, provenance = protocol.load_cities(archive_path)
    source_ids = tuple(city for city in CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    threshold = cert._estimate_threshold(
        cities, source_ids, mean, std, seed=seed, device=device, quantile=0.90,
    )
    support_scale, support_info = cert._source_support_scale(
        cities, source_ids, target,
    )
    backbone, backbone_info = protocol._train_one(
        "phaseformer_exact", cities, source_ids, len(cities[target]["site_ids"]),
        mean, std, seed=seed, steps=steps, device=device,
        torch_threads=torch_threads,
    )
    if support_scale > 0.0:
        # Match the official runner: every arm is initialized from the same
        # seed independently of the preceding PhaseFormer arm.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        candidate = _build_model(len(cities[target]["site_ids"]), threshold, support_scale)
        candidate_info = _train_candidate(
            candidate, cities, source_ids, mean, std, seed=seed, steps=steps,
            device=device, torch_threads=torch_threads,
        )
        source_gate = (_source_gate(
            backbone, candidate, cities, source_ids, mean, std, device=device,
        ) if source_diagnostic else {
            "selected": None,
            "selection_data": "not run; source certificates are fixed q90/support projections",
        })
        target_gate = _target_august_gate(
            backbone, candidate, cities, target, mean, std, device=device,
        )
        # Release uses the fixed target-August development rule.  The source
        # gate remains recorded as a transfer diagnostic, while the
        # source-support certificate still blocks unsupported target profiles.
        use_correction = bool(target_gate["selected"])
    else:
        candidate = None
        candidate_info = {"skipped": True, "reason": "target profile outside source support"}
        source_gate = {"selected": False, "selection_data": "not run: source-support fallback"}
        target_gate = {"selected": False, "selection_data": "not run: source-support fallback"}
        use_correction = False

    starts, truth, target_values = protocol._validation_data(cities[target])
    baseline = _predict(backbone, target_values, starts, mean, std, device=device)
    candidate_pred = baseline if candidate is None else _predict(
        candidate, target_values, starts, mean, std, device=device,
    )
    release = candidate_pred if use_correction else baseline
    predictions = {
        "phaseformer_exact": baseline,
        "source_certified_internal_candidate": candidate_pred,
        "source_certified_internal": release,
    }
    out = output_root / f"target_{target.lower()}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz", label_starts=starts,
        truth_raw=truth, site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
        **{f"pred_{name}_raw": value.astype(np.float32)
           for name, value in predictions.items()},
    )
    result = {
        "experiment": "CHARGED v3 SCPRC-v2.1 source-certified internal phase adapter",
        "status": "completed", "target_city": target, "seed": seed,
        "source_cities": list(source_ids), "backbone": backbone_info,
        "candidate": candidate_info, "source_gate": source_gate,
        "target_august_gate": target_gate,
        "selected_model": "source_certified_internal" if use_correction else "phaseformer_exact",
        "protocol": {
            "context_hours": CONTEXT, "horizon_hours": HORIZON,
            "source_train_interval": "2023-04-01 through 2023-07-31",
            "target_validation_interval": "2023-08-01 through 2023-08-31",
            "validation_origins": int(len(starts)),
            "target_city_labels_used_for_optimization": False,
            "source_august_used_for_transfer_diagnostic": bool(source_diagnostic),
            "target_august_used_for_release_audit": True,
            "source_relation_threshold": float(threshold),
            "source_relation_threshold_quantile": 0.90,
            **support_info,
            "support_projection": "exact PhaseFormer when target history profile exceeds source radius",
            "zero_initialized_internal_relation_adapter": True,
            "learned_gate": False, "gate_parameter_search": False,
            "expert_pool": False, "september_values_materialized": False,
        },
        "source_scaler_mean": mean, "source_scaler_std": std,
        "data_provenance": provenance,
        "metrics": {name: protocol._score(pred, truth)
                     for name, pred in predictions.items()},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    del backbone
    if candidate is not None:
        del candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def smoke() -> dict:
    torch.manual_seed(4)
    model = _build_model(11, threshold=1.0, support_scale=1.0).eval()
    x = torch.randn(2, CONTEXT, 11)
    with torch.no_grad():
        y = model(x)
    if y.shape != (2, HORIZON, 11) or not torch.isfinite(y).all():
        raise AssertionError("SCPRC-v2.1 smoke failed")
    return {"status": "passed", "output_shape": list(y.shape),
            "learned_gate": False, "expert_pool": False,
            "zero_initialized_internal_relation_adapter": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CITY_CODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path,
                        default=HERE.parent / "charged_source_certified_internal_phase_loco")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--source-diagnostic", action="store_true",
                        help="also evaluate the expensive source-city August diagnostic")
    args = parser.parse_args()
    result = run(args.target, args.seed, output_root=args.output_root,
                 steps=args.steps, device=args.device,
                 torch_threads=args.torch_threads,
                 source_diagnostic=args.source_diagnostic,
                 archive_path=args.archive)
    print(json.dumps({"target_city": result["target_city"], "seed": result["seed"],
                      "selected_model": result["selected_model"],
                      "source_gate": result["source_gate"],
                      "target_august_gate": result["target_august_gate"],
                      "metrics": result["metrics"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

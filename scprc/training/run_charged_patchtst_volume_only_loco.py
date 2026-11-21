#!/usr/bin/env python3
"""Fair volume-only official PatchTST control for the frozen SCPRC protocol.

The budget is counted in source minibatches: 2048 source minibatches, batch 4,
8192 effective windows. Five-source PCGrad aggregates one task gradient per
active source city, producing 410 AdamW parameter updates (the final round is
partial). Official PatchTST supports this through standard autograd. August
is retrospective and is never used for fitting, scaling, or adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
FROZEN_PROTOCOL_PATH = EXPERIMENTS / 'scprc_matched_protocol.json'
FROZEN_PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
_protocol_bytes = FROZEN_PROTOCOL_PATH.read_bytes()
if hashlib.sha256(_protocol_bytes).hexdigest() != FROZEN_PROTOCOL_SHA256:
    raise RuntimeError("frozen SCPRC protocol SHA-256 mismatch; refusing to run")
FROZEN_PROTOCOL = json.loads(_protocol_bytes)

for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as volume_protocol  # noqa: E402
from scprc.models import timemixer as _timemixer_namespace  # noqa: F401,E402
from scprc.baselines.patchtst_adapter import (  # noqa: E402
    PatchTSTUrbanEV,
    UPSTREAM_COMMIT,
)

CITY_CODES = tuple(FROZEN_PROTOCOL["data_split"]["cities"])
CONTEXT = int(FROZEN_PROTOCOL["input"]["context_hours"])
HORIZON = int(FROZEN_PROTOCOL["input"]["forecast_horizon_hours"])
TRAIN_END = 2928
VAL_END = 3672
SEEDS = tuple(int(value) for value in FROZEN_PROTOCOL["training_budget"]["seeds"])
SOURCE_MINIBATCHES = int(FROZEN_PROTOCOL["training_budget"]["source_minibatches_total"])
BATCH_SIZE = int(FROZEN_PROTOCOL["training_budget"]["batch_size_windows"])
EFFECTIVE_WINDOWS = int(FROZEN_PROTOCOL["training_budget"]["effective_source_windows"])
LEARNING_RATE = float(FROZEN_PROTOCOL["training_budget"]["learning_rate"])
WEIGHT_DECAY = float(FROZEN_PROTOCOL["training_budget"]["weight_decay"])
CONSISTENCY_WEIGHT = 0.15
CONSISTENCY_FACTORS = (2, 3, 4, 6, 12)


def _validate_frozen_protocol() -> None:
    expected = {
        "features": ["volume"],
        "target": "volume",
        "context_hours": 168,
        "forecast_horizon_hours": 24,
        "duration_input": False,
        "source_minibatches_total": 2048,
        "batch_size_windows": 4,
        "effective_source_windows": 8192,
    }
    observed = {**FROZEN_PROTOCOL["input"], **FROZEN_PROTOCOL["training_budget"]}
    for key, value in expected.items():
        if observed.get(key) != value:
            raise RuntimeError(
                f"frozen protocol mismatch for {key}: {observed.get(key)!r} != {value!r}"
            )


_validate_frozen_protocol()


class VolumeOnlyPatchTST(nn.Module):
    """Pinned official PatchTST with one historical volume channel."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PatchTSTUrbanEV(
            seq_len=CONTEXT,
            pred_len=HORIZON,
            channels=1,
            patch_len=16,
            stride=8,
            d_model=32,
            d_ff=64,
            e_layers=1,
            n_heads=4,
            dropout=0.0,
            fc_dropout=0.0,
            head_dropout=0.0,
            revin=True,
            decomposition=False,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (CONTEXT, 1):
            raise ValueError(f"expected [batch,{CONTEXT},1], got {tuple(history.shape)}")
        forecast = self.backbone(history)
        if tuple(forecast.shape[1:]) != (HORIZON, 1):
            raise RuntimeError(f"unexpected PatchTST output shape {tuple(forecast.shape)}")
        return forecast[..., 0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_summary(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


def _objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    point = (prediction - target).square().mean()
    terms = []
    for factor in CONSISTENCY_FACTORS:
        if prediction.shape[1] % factor:
            continue
        pred_scale = prediction.reshape(
            prediction.shape[0], prediction.shape[1] // factor, factor, prediction.shape[2]
        ).mean(dim=2)
        target_scale = target.reshape(
            target.shape[0], target.shape[1] // factor, factor, target.shape[2]
        ).mean(dim=2)
        terms.append((pred_scale - target_scale).square().mean())
    return point + CONSISTENCY_WEIGHT * torch.stack(terms).mean() if terms else point


def _global_dot(left: tuple[torch.Tensor | None, ...],
                right: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    products = [(first * second).sum() for first, second in zip(left, right)
                if first is not None and second is not None]
    if not products:
        return torch.zeros((), device=next(
            (gradient.device for gradient in (*left, *right) if gradient is not None),
            torch.device("cpu"),
        ))
    return torch.stack(products).sum()


def _project_conflicts(task_grads: list[tuple[torch.Tensor | None, ...]],
                       order: np.ndarray) -> tuple[list[tuple[torch.Tensor | None, ...]], int, int]:
    original = task_grads
    projected = [[gradient.clone() if gradient is not None else None for gradient in task]
                 for task in original]
    conflicts = 0
    pairs = 0
    for task_index in order:
        for other_index in order:
            if task_index == other_index:
                continue
            dot = _global_dot(tuple(projected[task_index]), original[other_index])
            norm = _global_dot(original[other_index], original[other_index]).clamp_min(1e-20)
            pairs += 1
            if dot.item() < 0.0:
                conflicts += 1
                ratio = dot / norm
                for parameter_index, other_gradient in enumerate(original[other_index]):
                    current = projected[task_index][parameter_index]
                    if current is not None and other_gradient is not None:
                        current.add_(other_gradient, alpha=-float(ratio.item()))
    return [tuple(task) for task in projected], conflicts, pairs


def _code_provenance() -> dict[str, str]:
    paths = [
        HERE,
        FROZEN_PROTOCOL_PATH,
        EXPERIMENTS / "training_protocol.py",
        ROOT / "scprc/baselines/patchtst_adapter.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/models/PatchTST.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/PatchTST_layers.py",
        ROOT / "scprc/vendor/PatchTST/PatchTST_supervised/layers/RevIN.py",
    ]
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths if path.is_file()}


def _hardware(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    try:
        payload["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        payload["nvidia_smi"] = []
    return payload


def write_metadata(output_dir: Path, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(FROZEN_PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "protocol_id": FROZEN_PROTOCOL["protocol_id"],
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "runner": str(HERE.relative_to(ROOT)),
        "runner_sha256": _sha256(HERE),
        "hardware": _hardware(device),
        "code_provenance": _code_provenance(),
        "recorded_at_unix": time.time(),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _fit_model(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...],
               mean: float, std: float, *, seed: int, device: str,
               torch_threads: int) -> tuple[VolumeOnlyPatchTST, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = VolumeOnlyPatchTST().float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    amp_enabled = False
    starts = volume_protocol._train_starts()
    rng = np.random.default_rng(seed)
    minibatches_by_city = {city: 0 for city in source_ids}
    losses: list[float] = []
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    base, remainder = divmod(SOURCE_MINIBATCHES, len(source_ids))
    minibatches_per_city = {
        city: base + (index < remainder) for index, city in enumerate(source_ids)
    }
    parameter_updates = max(minibatches_per_city.values())
    update_order_rng = np.random.default_rng(seed ^ 0x5C2C)
    conflict_pairs = 0
    gradient_pairs = 0
    started = time.perf_counter()
    model.train()
    for update_index in range(parameter_updates):
        active_cities = [city for city in source_ids
                         if update_index < minibatches_per_city[city]]
        task_grads = []
        for city_id in active_cities:
            x, y = volume_protocol._make_batch(
                cities[city_id]["values"][:TRAIN_END], starts, mean, std, rng, BATCH_SIZE
            )
            # [B,T,N] -> [B*N,T,1], making sites independent samples as in the
            # official channel-independent PatchTST implementation.
            batch, steps, nodes = x.shape
            model_x = x.permute(0, 2, 1).contiguous().reshape(batch * nodes, steps, 1).to(device)
            model_y = y.permute(0, 2, 1).contiguous().reshape(batch * nodes, HORIZON).to(device)
            prediction = model(model_x)
            loss = _objective(
                prediction.unsqueeze(-1), model_y.unsqueeze(-1)
            )
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite output/loss at parameter update {update_index + 1}, {city_id}"
                )
            gradients = torch.autograd.grad(
                loss, parameters, retain_graph=False, create_graph=False, allow_unused=True,
            )
            if not all(gradient is None or torch.isfinite(gradient).all()
                       for gradient in gradients):
                raise FloatingPointError(
                    f"non-finite gradient at parameter update {update_index + 1}, {city_id}"
                )
            task_grads.append(gradients)
            minibatches_by_city[city_id] += 1
            losses.append(float(loss.detach().cpu()))
        if len(task_grads) > 1:
            order = update_order_rng.permutation(len(task_grads))
            projected, conflicts, pairs = _project_conflicts(task_grads, order)
            conflict_pairs += conflicts
            gradient_pairs += pairs
        else:
            projected = task_grads
        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            available = [task[parameter_index] for task in projected
                         if task[parameter_index] is not None]
            if available:
                parameter.grad = torch.stack(available, dim=0).mean(dim=0)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in parameters):
            raise FloatingPointError(f"non-finite projected gradient at update {update_index + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, {
        "source_minibatches_total": SOURCE_MINIBATCHES,
        "effective_source_windows": SOURCE_MINIBATCHES * BATCH_SIZE,
        "parameter_updates": int(parameter_updates),
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "parameter_update_schedule": "one AdamW update after PCGrad aggregation across active source-city minibatches",
        "source_minibatches_by_city": minibatches_by_city,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size_windows": BATCH_SIZE,
        "loss": "standardized volume MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "gradient_aggregation": "PCGrad pairwise projection on conflicting source-city gradients",
        "gradient_projection_order": "seeded shuffle per parameter update",
        "source_gradient_conflict_pairs": int(conflict_pairs),
        "source_gradient_pair_count": int(gradient_pairs),
        "source_gradient_conflict_fraction": conflict_pairs / gradient_pairs if gradient_pairs else 0.0,
        "mixed_precision_enabled": amp_enabled,
        "gradient_clip_norm": 1.0,
        "seed": int(seed),
        "mean_training_loss": float(np.mean(losses)),
        "last_training_loss": float(losses[-1]),
        "training_seconds": float(time.perf_counter() - started),
        "parameters": _parameter_summary(model),
    }


def _predict(model: VolumeOnlyPatchTST, city: dict[str, Any], mean: float,
             std: float, *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(TRAIN_END, VAL_END - HORIZON + 1, dtype=np.int64)
    values = city["values"]
    predictions, truths = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            picks = starts[offset:offset + BATCH_SIZE]
            x = np.stack([values[start - CONTEXT:start] for start in picks]).astype(np.float32)
            y = np.stack([values[start:start + HORIZON] for start in picks]).astype(np.float32)
            x = (x - mean) / std
            batch, steps, nodes = x.shape
            model_x = torch.from_numpy(x).permute(0, 2, 1).contiguous().reshape(
                batch * nodes, steps, 1
            ).to(device)
            pred_scaled = model(model_x).reshape(batch, nodes, HORIZON).transpose(1, 2)
            predictions.append((pred_scaled.cpu().numpy() * std + mean).astype(np.float32))
            truths.append(y)
    return np.concatenate(predictions), np.concatenate(truths), starts


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error)))}


def run_fold(*, target: str, seed: int, output_dir: Path, device: str,
             torch_threads: int, force: bool = False) -> Path:
    target = target.upper()
    if target not in CITY_CODES:
        raise ValueError(f"target must be one of {CITY_CODES}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    fold_dir = output_dir / f"target_{target.lower()}_seed{seed}"
    result_path = fold_dir / "result.json"
    if result_path.exists() and not force:
        raise FileExistsError(f"existing fold {result_path}; use --force to replace it")
    fold_dir.mkdir(parents=True, exist_ok=True)
    cities, data_provenance = volume_protocol.load_cities()
    source_ids = tuple(city for city in CITY_CODES if city != target)
    mean, std = volume_protocol.source_scaler(cities, source_ids)
    model, training = _fit_model(
        cities, source_ids, mean, std, seed=seed, device=device,
        torch_threads=torch_threads,
    )
    prediction, truth, starts = _predict(model, cities[target], mean, std, device=device)
    prediction_file = fold_dir / "predictions.npz"
    np.savez_compressed(
        prediction_file,
        label_starts=starts,
        label_timestamps=np.asarray([str(cities[target]["times"][int(start)]) for start in starts]),
        site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
        truth_volume_raw=truth,
        prediction_patchtst_volume_only_raw=prediction,
    )
    checkpoint_file = fold_dir / "checkpoint.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint_file)
    result: dict[str, Any] = {
        "experiment": "CHARGED SCPRC matched official PatchTST volume-only LOCO",
        "status": "completed",
        "retrospective": True,
        "target_city": target,
        "source_cities": list(source_ids),
        "seed": int(seed),
        "metrics_raw_volume": _metrics(prediction, truth),
        "validation_windows": int(len(starts)),
        "site_count": int(len(cities[target]["site_ids"])),
        "prediction_file": str(prediction_file),
        "checkpoint_file": str(checkpoint_file),
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "training": training,
        "protocol": {
            "protocol_id": FROZEN_PROTOCOL["protocol_id"],
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "features": ["volume"],
            "target": "volume",
            "duration_input": False,
            "context_hours": CONTEXT,
            "horizon_hours": HORIZON,
            "source_minibatches_total": SOURCE_MINIBATCHES,
            "batch_size_windows": BATCH_SIZE,
            "effective_source_windows": EFFECTIVE_WINDOWS,
            "parameter_updates": training["parameter_updates"],
            "source_schedule": "round-robin over five source cities",
            "source_only_scaler": True,
            "target_adaptation_steps": 0,
            "evaluation_window": "August 2023, 721 rolling origins; retrospective only",
            "september_values_materialized": False,
            "blind_confirmation_claim": False,
        },
        "architecture": {
            "name": "pinned official PatchTST, volume-only",
            "implementation": "PatchTST_supervised pinned adapter",
            "upstream_commit": UPSTREAM_COMMIT,
            "input_channels": 1,
            "input_variables": ["volume"],
            "output_channels": 1,
            "patch_len": 16,
            "stride": 8,
            "d_model": 32,
            "d_ff": 64,
            "e_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "revin": True,
            "channel_independence": True,
            "cross_variable_fusion": "not applicable: duration is excluded",
            "multi_resolution_consistency": "external objective, factors 2,3,4,6,12, weight 0.15",
            "pcgrad": "applied across active source-city minibatches; 410 AdamW updates under 2048-minibatch budget",
        },
        "data_provenance": data_provenance,
        "code_provenance": _code_provenance(),
        "hardware": _hardware(device),
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result_path


def _pcgrad_smoke() -> dict[str, Any]:
    task_grads = [
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-1.0, 1.0]),),
    ]
    projected, conflicts, pairs = _project_conflicts(task_grads, np.asarray([0, 1]))
    aggregate = torch.stack([task[0] for task in projected]).mean(dim=0)
    if conflicts != 2 or pairs != 2 or not torch.isfinite(aggregate).all():
        raise AssertionError("PCGrad projection smoke failed")
    return {"status": "passed", "conflicting_pairs": conflicts, "pairs": pairs,
            "aggregate_gradient": aggregate.tolist()}


def cpu_smoke() -> dict[str, Any]:
    torch.manual_seed(0)
    model = VolumeOnlyPatchTST().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    task_grads = []
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    for task_index in range(2):
        x = torch.randn(4, CONTEXT, 1, dtype=torch.float32)
        target = torch.randn(4, HORIZON, 1, dtype=torch.float32)
        prediction = model(x)
        loss = _objective(prediction.unsqueeze(-1), target)
        if tuple(prediction.shape) != (4, HORIZON) or not torch.isfinite(loss):
            raise AssertionError("volume-only PatchTST forward/loss smoke failed")
        gradients = torch.autograd.grad(loss, trainable_parameters, allow_unused=True)
        task_grads.append(gradients)
    projected, conflicts, pairs = _project_conflicts(task_grads, np.asarray([0, 1]))
    optimizer.zero_grad(set_to_none=True)
    for parameter_index, parameter in enumerate(trainable_parameters):
        available = [task[parameter_index] for task in projected if task[parameter_index] is not None]
        if available:
            parameter.grad = torch.stack(available).mean(dim=0)
    optimizer.step()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in trainable_parameters):
        raise AssertionError("volume-only PatchTST PCGrad smoke failed")
    return {
        "status": "passed",
        "output_shape": [4, HORIZON],
        "parameter_summary": _parameter_summary(model),
        "source_minibatches_total": SOURCE_MINIBATCHES,
        "effective_source_windows": EFFECTIVE_WINDOWS,
        "parameter_updates_pcgrad": (SOURCE_MINIBATCHES + len(CITY_CODES) - 2) // (len(CITY_CODES) - 1),
        "consistency_weight": CONSISTENCY_WEIGHT,
        "pcgrad_projection_smoke": {
            "status": "passed",
            "source_tasks": len(task_grads),
            "conflicting_pairs": conflicts,
            "gradient_pairs": pairs,
            "optimizer_step": "passed",
        },
        "official_commit": UPSTREAM_COMMIT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CITY_CODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "charged_patchtst_volume_only_loco_20261023")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(cpu_smoke(), indent=2, ensure_ascii=False))
        return 0
    if args.write_metadata:
        write_metadata(args.output_dir, args.device)
        print(args.output_dir)
        return 0
    if args.target is None or args.seed is None:
        parser.error("--target and --seed are required unless --smoke-only or --write-metadata is used")
    result = run_fold(
        target=args.target, seed=args.seed, output_dir=args.output_dir,
        device=args.device, torch_threads=args.torch_threads, force=args.force,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

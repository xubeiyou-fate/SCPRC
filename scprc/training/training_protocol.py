#!/usr/bin/env python3
"""Source-only CHARGED LOCO for phase-conditioned station relations.

This runner consumes only the April--August prefix of CHARGED v3 volume.csv.
For each target city, all optimization batches come from the other five
cities through July. The held-out city's August labels are used only for
paired development evaluation; September rows are neither parsed nor stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
REPRO = ROOT / "scprc"
EXTERNAL = ROOT / 'scprc/data_adapters'
CHARGED = EXTERNAL / "charged"
for _path in (ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scprc.data_adapters.charged.city_adapter import (  # noqa: E402
    ARCHIVE, ARCHIVE_MD5, CITY_CODES, DATASET_DOI, FIRST_HOUR, LICENSE,
)
from scprc.baselines.phaseformer_adapter import PhaseFormerUrbanEV, UPSTREAM_COMMIT  # noqa: E402
from scprc.training.phase_relations import (  # noqa: E402
    InputPhaseRelationPhaseFormer,
)
from scprc.training.relation_attenuation import (  # noqa: E402
    ScaleCappedPhaseRelationPhaseFormer,
    ReliabilityCappedPhaseRelationPhaseFormer,
)

CONTEXT = 168
HORIZON = 24
TRAIN_END = 2928
VAL_END = 3672
STEPS = 2048
BATCH_SIZE = 4
LR = 1e-3
CONSISTENCY_WEIGHT = 0.15
CONSISTENCY_FACTORS = (2, 3, 4, 6, 12)
MODEL_NAMES = ("phaseformer_exact", "phase_relation", "phase_agnostic_daily",
               "phase_relation_scale_capped", "phase_relation_reliability_capped")


class VariableSitePhaseFormer(torch.nn.Module):
    """Official PhaseFormer core with shared weights over a variable site axis."""

    def __init__(self, n_sites: int):
        super().__init__()
        adapter = PhaseFormerUrbanEV(
            seq_len=CONTEXT, pred_len=HORIZON, channels=n_sites, period_len=24,
            latent_dim=32, phase_layers=2, routers=1, heads=8,
            dropout=0.1, pos_embed=True,
        )
        self.network = adapter.network

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != CONTEXT:
            raise ValueError(f"expected [batch,{CONTEXT},sites], got {tuple(history.shape)}")
        prediction, _, _ = self.network(history)
        if prediction.shape[:2] != (history.shape[0], HORIZON) or prediction.shape[2] != history.shape[2]:
            raise RuntimeError(f"unexpected PhaseFormer output shape {tuple(prediction.shape)}")
        return prediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_prefix_sha256(archive: zipfile.ZipFile, member: str, rows: int) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        for _ in range(rows + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"{member} ended before {rows} requested data rows")
            digest.update(line)
    return digest.hexdigest()


def _read_volume_prefix(archive: zipfile.ZipFile, city: str) -> dict[str, Any]:
    member = f"{city}/volume.csv"
    try:
        with archive.open(member) as stream:
            frame = pd.read_csv(stream, nrows=VAL_END)
    except KeyError as error:
        raise FileNotFoundError(member) from error
    if frame.shape[0] != VAL_END or frame.shape[1] < 2:
        raise ValueError(f"{member} must have timestamp plus sites for {VAL_END} rows")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range(FIRST_HOUR, periods=VAL_END, freq="h")
    site_ids = tuple(str(value) for value in frame.columns[1:])
    if not times.equals(expected):
        raise ValueError(f"{member} timestamps do not match the approved April--August prefix")
    if not site_ids or len(set(site_ids)) != len(site_ids) or not all(site_ids):
        raise ValueError(f"{member} has empty or duplicate site IDs")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.shape != (VAL_END, len(site_ids)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{member} contains invalid values")
    return {
        "city_id": city,
        "times": times,
        "site_ids": site_ids,
        "values": values,
        "source_member": member,
        "prefix_sha256": _member_prefix_sha256(archive, member, VAL_END),
        "rows_materialized": VAL_END,
        "last_timestamp_materialized": str(times[-1]),
    }


def load_cities(archive_path: Path = ARCHIVE) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cities = {}
    with zipfile.ZipFile(archive_path) as archive:
        for city in CITY_CODES:
            cities[city] = _read_volume_prefix(archive, city)
    return cities, {
        "dataset": "CHARGED v3 standard hourly archive; remove_zero=False",
        "dataset_doi": DATASET_DOI,
        "license": LICENSE,
        "archive_path": str(archive_path),
        "archive_md5_published": ARCHIVE_MD5,
        "archive_size_bytes": archive_path.stat().st_size,
        "target": "volume only (kWh per site per local hour)",
        "rows_materialized_per_city": VAL_END,
        "last_timestamp_materialized": "2023-08-31 23:00:00",
        "september_values_materialized": False,
        "city_prefix_sha256": {city: value["prefix_sha256"] for city, value in cities.items()},
    }


def source_scaler(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...]) -> tuple[float, float]:
    means, second_moments = [], []
    for city in source_ids:
        prefix = cities[city]["values"][:TRAIN_END].astype(np.float64)
        means.append(float(prefix.mean()))
        second_moments.append(float(np.square(prefix).mean()))
    mean = float(np.mean(means))
    std = float(np.sqrt(max(float(np.mean(second_moments)) - mean * mean, 0.0)))
    return mean, max(std, 1e-6)


def _train_starts() -> np.ndarray:
    return np.arange(CONTEXT, TRAIN_END - HORIZON + 1, dtype=np.int64)


def _make_batch(values: np.ndarray, starts: np.ndarray, mean: float, std: float,
                rng: np.random.Generator, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    picks = rng.choice(starts, size=batch_size, replace=True)
    xs = np.stack([values[start - CONTEXT:start] for start in picks]).astype(np.float32)
    ys = np.stack([values[start:start + HORIZON] for start in picks]).astype(np.float32)
    return torch.from_numpy((xs - mean) / std), torch.from_numpy((ys - mean) / std)


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


def _build_model(name: str, n_sites: int):
    if name == "phaseformer_exact":
        return VariableSitePhaseFormer(n_sites)
    kwargs = dict(channels=n_sites, seq_len=CONTEXT, pred_len=HORIZON, period_len=24,
                  latent_dim=32, phase_layers=2, routers=1, heads=8,
                  dropout=0.1, pos_embed=True)
    if name == "phase_relation_scale_capped":
        return ScaleCappedPhaseRelationPhaseFormer(
            **kwargs, relation_mode="phase", relation_hidden=16, relation_rank=8
        )
    if name == "phase_relation_reliability_capped":
        return ReliabilityCappedPhaseRelationPhaseFormer(
            **kwargs, relation_mode="phase", relation_hidden=16, relation_rank=8
        )
    mode = "phase" if name == "phase_relation" else "phase_agnostic_daily"
    return InputPhaseRelationPhaseFormer(**kwargs, relation_mode=mode,
                                         relation_hidden=16, relation_rank=8)


def _predict(model, values: np.ndarray, starts: np.ndarray, mean: float, std: float,
             *, device: str, batch_size: int = 4) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            x = np.stack([values[s - CONTEXT:s] for s in batch_starts]).astype(np.float32)
            x = (x - mean) / std
            prediction = model(torch.from_numpy(x).to(device))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("non-finite validation forecast")
            chunks.append(prediction.cpu().numpy())
    return np.concatenate(chunks, axis=0) * std + mean


def _score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error)))}


def _validation_data(city: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(TRAIN_END, VAL_END - HORIZON + 1, dtype=np.int64)
    values = city["values"]
    truth = np.stack([values[s:s + HORIZON] for s in starts]).astype(np.float32)
    return starts, truth, values


def _train_one(name: str, cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...],
               target_sites: int, mean: float, std: float, *, seed: int,
               steps: int, device: str, torch_threads: int = 2) -> tuple[Any, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = _build_model(name, target_sites).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    amp_enabled = torch.device(device).type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    starts = _train_starts()
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    started = time.perf_counter()
    model.train()
    for update in range(steps):
        city_id = source_ids[update % len(source_ids)]
        x, y = _make_batch(cities[city_id]["values"][:TRAIN_END], starts, mean, std,
                           rng, BATCH_SIZE)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            prediction = model(x)
            loss = _objective(prediction.float(), y)
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training output/loss in {name} at update {update + 1}")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        updates_by_city[city_id] += 1
    return model, {
        "model": name,
        "seed": seed,
        "steps": steps,
        "batch_size_windows": BATCH_SIZE,
        "torch_threads": int(torch_threads),
        "source_cities": list(source_ids),
        "source_optimizer_updates_by_city": updates_by_city,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": 1e-4,
        "loss": "standardized MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": time.perf_counter() - started,
    }


def run(target: str, seed: int, *, output_root: Path, steps: int = STEPS,
        device: str = "cuda", archive_path: Path = ARCHIVE,
        model_names: tuple[str, ...] = MODEL_NAMES) -> dict[str, Any]:
    target = target.upper()
    if target not in CITY_CODES:
        raise ValueError(f"target must be one of {CITY_CODES}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if steps == STEPS and seed not in (13, 17, 29):
        raise ValueError("the locked full-budget experiment uses seeds 13, 17, and the approved robustness seed 29")
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError(f"model_names must be a nonempty subset of {MODEL_NAMES}")
    cities, data_provenance = load_cities(archive_path)
    source_ids = tuple(city for city in CITY_CODES if city != target)
    mean, std = source_scaler(cities, source_ids)
    starts, truth, target_values = _validation_data(cities[target])
    predictions = {}
    training = {}
    for name in model_names:
        model, info = _train_one(name, cities, source_ids, len(cities[target]["site_ids"]),
                                 mean, std, seed=seed, steps=steps, device=device)
        pred_scaled = _predict(model, target_values, starts, mean, std, device=device)
        if pred_scaled.shape != truth.shape:
            raise RuntimeError(f"unexpected prediction shape {pred_scaled.shape}, expected {truth.shape}")
        predictions[name] = pred_scaled.astype(np.float32)
        training[name] = {**info, "validation_raw_volume": _score(pred_scaled, truth)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = output_root / f"target_{target.lower()}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz", label_starts=starts,
        label_timestamps=np.asarray([str(cities[target]["times"][int(s)]) for s in starts]),
        site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
        truth_raw=truth,
        **{f"pred_{name}_raw": value for name, value in predictions.items()},
    )
    result = {
        "experiment": "CHARGED v3 source-only six-city LOCO: PhaseFormer input-phase historical station relation",
        "status": "completed",
        "target_city": target,
        "source_cities": list(source_ids),
        "model_names": list(model_names),
        "seed": seed,
        "protocol_amendment": "crosscity_phase_relation_protocol_amendment_2026-10-19.md",
        "protocol": {
            "context_hours": CONTEXT,
            "horizon_hours": HORIZON,
            "source_train_interval": "2023-04-01 00:00 through 2023-07-31 23:00",
            "source_train_end_exclusive_index": TRAIN_END,
            "target_validation_interval": "2023-08-01 00:00 through 2023-08-31 23:00",
            "target_validation_end_exclusive_index": VAL_END,
            "validation_origins": int(len(starts)),
            "target_city_labels_used_for_optimization": False,
            "target_city_statistics_used_for_scaling": False,
            "target_city_context_used_at_forecast_origins": True,
            "source_scaler": "equal-city-weight mean/std from five source cities over indices [0,2928)",
            "rows_materialized_per_city": VAL_END,
            "september_values_materialized": False,
            "site_count_target": int(len(cities[target]["site_ids"])),
            "steps_per_model": steps,
            "full_protocol_budget": steps == STEPS and seed in (13, 17, 29),
        },
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "data_provenance": data_provenance,
        "code_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (HERE, HERE.parent / "phase_relations.py",
                         HERE.parent / "crosscity_phase_relation_protocol_amendment_2026-10-19.md",
                         CHARGED / "city_adapter.py", EXTERNAL / "cross_city_adapter.py",
                         REPRO / "baselines" / "phaseformer_adapter.py")
            if path.is_file()
        },
        "phaseformer_upstream_commit": UPSTREAM_COMMIT,
        "training_and_metrics": training,
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def smoke() -> dict[str, Any]:
    torch.manual_seed(4)
    model = InputPhaseRelationPhaseFormer(channels=5, dropout=0.0)
    model.eval()
    x = torch.randn(2, CONTEXT, 11)
    with torch.no_grad():
        y = model(x)
    if y.shape != (2, HORIZON, 11) or not torch.isfinite(y).all():
        raise AssertionError("variable-site smoke failed")
    starts = _train_starts()
    if starts[0] != CONTEXT or starts[-1] + HORIZON != TRAIN_END:
        raise AssertionError("source windows do not end at the training boundary")
    val_starts = np.arange(TRAIN_END, VAL_END - HORIZON + 1)
    if len(val_starts) != 721 or val_starts[-1] + HORIZON != VAL_END:
        raise AssertionError("August validation windows are not bounded correctly")
    return {"status": "passed", "smoke_only": True,
            "source_train_windows": int(len(starts)), "target_validation_origins": int(len(val_starts)),
            "variable_site_count": 11, "output_shape": list(y.shape),
            "archive_read": False, "september_read": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CITY_CODES)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=HERE.parent / "charged_input_phase_relation_loco_runs")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(smoke(), sort_keys=True))
        return 0
    if args.target is None:
        parser.error("--target is required unless --smoke-only is set")
    result = run(args.target, args.seed, output_root=args.output_root, steps=args.steps,
                 device=args.device, archive_path=args.archive,
                 model_names=tuple(args.models))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

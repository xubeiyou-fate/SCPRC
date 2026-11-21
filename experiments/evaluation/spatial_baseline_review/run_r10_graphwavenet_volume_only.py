#!/usr/bin/env python3
"""Run the R10 volume-only fixed-distance Graph WaveNet CHARGED baseline.

This runner intentionally has no learned node-ID adjacency.  It is a real
static spatial baseline and must not be labelled adaptive Graph WaveNet.
All output is written under the caller-provided output root, which should be
inside the R10 directory.
"""
from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
REPRO = ROOT / "scprc"
EXTERNAL = ROOT / 'scprc/data_adapters'
for path in (EXPERIMENTS, REPRO, EXTERNAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.run_charged_covariate_hptn_phaseformer_loco as charged_protocol  # noqa: E402
from scprc.baselines.graphwavenet_charged_adapter import (  # noqa: E402
    ChargedFixedGraphWaveNet,
    FixedSupport,
    distance_supports,
)


CITIES = tuple(charged_protocol.CITY_CODES)
SEEDS = (13, 17, 29)
CONTEXT = int(charged_protocol.CONTEXT)
HORIZON = int(charged_protocol.HORIZON)
TRAIN_END = int(charged_protocol.TRAIN_END)
VAL_END = int(charged_protocol.VAL_END)
ARCHIVE = Path(charged_protocol.ARCHIVE)
STEPS = 2048
BATCH_WINDOWS = 4
MICROBATCH_WINDOWS = 1
LEARNING_RATE = 1e-3
TOP_K = 8
UPSTREAM_COMMIT = "6b162e80c59a1d494809252eca055cff93dc66b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def source_code_hashes() -> dict[str, str]:
    paths = [
        HERE,
        REPRO / "optional_baselines/graphwavenet_charged_adapter.py",
        REPRO / "vendor/Graph-WaveNet/model.py",
        EXPERIMENTS / "run_charged_covariate_hptn_phaseformer_loco.py",
        EXTERNAL / "cross_city_adapter.py",
        EXTERNAL / "charged/city_adapter.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.is_file()}


def load_cities_and_supports() -> tuple[dict[str, dict[str, Any]], dict[str, tuple[FixedSupport, FixedSupport]], dict[str, Any]]:
    """Load only the volume channel while hashing all consumed provenance."""
    cities, data_provenance = charged_protocol.load_feature_arrays()
    supports: dict[str, tuple[FixedSupport, FixedSupport]] = {}
    graph_hashes: dict[str, str] = {}
    for city in CITIES:
        values = np.asarray(cities[city]["values"], dtype=np.float32)
        if values.ndim != 3 or values.shape[2] < 1:
            raise ValueError(f"{city}: expected [time, station, variable] values")
        # Keep only volume.  This is the protocol-critical distinction from
        # the historical volume+duration Graph WaveNet run.
        cities[city] = dict(cities[city])
        cities[city]["values"] = values[..., :1]
        with zipfile.ZipFile(ARCHIVE) as archive:
            member = f"{city}/distance.csv"
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, index_col=0)
        expected = [str(site) for site in cities[city]["node_ids"]]
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if not set(expected).issubset(set(frame.index)) or not set(expected).issubset(set(frame.columns)):
            raise ValueError(f"{member} does not cover all aligned station ids")
        distance = frame.loc[expected, expected].to_numpy(dtype=np.float64)
        if not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError(f"{member} contains invalid distances")
        if not np.allclose(distance, distance.T, atol=1e-5, rtol=1e-5):
            raise ValueError(f"{member} is not symmetric")
        supports[city] = distance_supports(distance, top_k=TOP_K, sigma=None)
        graph_hashes[city] = hash_array(distance)
    graph = {
        "distance_source": "CHARGED v3 distance.csv aligned to volume station order",
        "distance_sha256_by_city": graph_hashes,
        "graph_rule": "symmetrized top-8 nearest neighbors plus self loop; Gaussian weights",
        "top_k": TOP_K,
        "sigma_rule": "city-specific median positive inter-site distance",
        "learned_node_embeddings": False,
        "addaptadj": False,
        "target_graph_access": "static target geometry used in target forward pass",
        "target_graph_is_transductive": True,
    }
    return cities, supports, {"data": data_provenance, "graph": graph}


def source_only_volume_scaler(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...]) -> tuple[float, float]:
    means = []
    second_moments = []
    for city in source_ids:
        prefix = cities[city]["values"][:TRAIN_END, ..., 0].astype(np.float64)
        means.append(prefix.mean())
        second_moments.append(np.square(prefix).mean())
    mean = float(np.mean(means))
    std = float(np.sqrt(max(np.mean(second_moments) - mean * mean, 0.0)))
    return mean, max(std, 1e-6)


def starts(label_start: int, label_end: int) -> np.ndarray:
    first = max(int(label_start), CONTEXT)
    last = int(label_end) - HORIZON
    return np.arange(first, last + 1, dtype=np.int64) if last >= first else np.empty(0, dtype=np.int64)


def batch(values: np.ndarray, labels: np.ndarray, mean: float, std: float,
          rng: np.random.Generator, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    picks = rng.choice(labels, size=count, replace=True)
    x = np.stack([values[s - CONTEXT:s] for s in picks]).astype(np.float32)
    x = x.transpose(0, 3, 2, 1)
    x = (x - mean) / std
    y = np.stack([values[s:s + HORIZON, :, 0] for s in picks]).astype(np.float32)
    y = (y - mean) / std
    return torch.from_numpy(x), torch.from_numpy(y)


def fit(cities: dict[str, dict[str, Any]], supports: dict[str, tuple[FixedSupport, FixedSupport]],
        source_ids: tuple[str, ...], mean: float, std: float, *, seed: int,
        steps: int, device: str, torch_threads: int) -> tuple[ChargedFixedGraphWaveNet, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    model = ChargedFixedGraphWaveNet(
        channels=1, seq_len=CONTEXT, pred_len=HORIZON,
        hidden=32, blocks=1, layers=8, dropout=0.3,
    ).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    labels = starts(0, TRAIN_END)
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    losses: list[float] = []
    started = time.perf_counter()
    for update in range(int(steps)):
        city = source_ids[update % len(source_ids)]
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        sampled = 0
        while sampled < BATCH_WINDOWS:
            count = min(MICROBATCH_WINDOWS, BATCH_WINDOWS - sampled)
            x, y = batch(cities[city]["values"], labels, mean, std, rng, count)
            x, y = x.to(device), y.to(device)
            prediction = model(x, supports[city])
            if prediction.shape != y.shape:
                raise RuntimeError(f"prediction {tuple(prediction.shape)} != target {tuple(y.shape)}")
            loss = F.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {update}")
            (loss * (count / BATCH_WINDOWS)).backward()
            update_loss += float(loss.detach().cpu()) * count / BATCH_WINDOWS
            sampled += count
        optimizer.step()
        updates_by_city[city] += 1
        losses.append(update_loss)
    return model, {
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "standardized source-only volume MSE",
        "source_optimizer_updates_total": int(steps),
        "source_optimizer_updates_by_city": updates_by_city,
        "batch_windows_per_update": BATCH_WINDOWS,
        "microbatch_windows": MICROBATCH_WINDOWS,
        "target_adaptation_steps": 0,
        "mean_training_loss": float(np.mean(losses)),
        "last_training_loss": float(losses[-1]),
        "training_seconds": float(time.perf_counter() - started),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def predict(model: ChargedFixedGraphWaveNet, city: dict[str, Any], graph: tuple[FixedSupport, FixedSupport],
            mean: float, std: float, *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = starts(TRAIN_END, VAL_END)
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(labels), 1):
            s = int(labels[offset])
            x = np.asarray(city["values"][s - CONTEXT:s], dtype=np.float32)[None, ...]
            x = x.transpose(0, 3, 2, 1)
            x = (x - mean) / std
            output = model(torch.from_numpy(x).to(device), graph).cpu().numpy()
            predictions.append(output)
            truths.append(city["values"][s:s + HORIZON, :, 0][None, ...].astype(np.float32))
    pred_scaled = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truths, axis=0)
    return (pred_scaled * std + mean).astype(np.float32), truth, labels


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error)))}


def protocol_payload() -> dict[str, Any]:
    return json.loads((HERE.parent / "protocol.json").read_text(encoding="utf-8"))


def run_fold(*, target: str, seed: int, output_root: Path, steps: int = STEPS,
             device: str = "cuda", torch_threads: int = 2, force: bool = False) -> dict[str, Any]:
    target = str(target).upper()
    if target not in CITIES or int(seed) not in SEEDS:
        raise ValueError(f"target must be one of {CITIES}; seed must be one of {SEEDS}")
    fold = output_root / f"target_{target.lower()}_seed{int(seed)}"
    result_path = fold / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    fold.mkdir(parents=True, exist_ok=True)
    cities, supports, provenance = load_cities_and_supports()
    source_ids = tuple(city for city in CITIES if city != target)
    mean, std = source_only_volume_scaler(cities, source_ids)
    model, train_info = fit(
        cities, supports, source_ids, mean, std, seed=int(seed), steps=steps,
        device=device, torch_threads=torch_threads,
    )
    prediction, truth, label_starts = predict(model, cities[target], supports[target], mean, std, device=device)
    prediction_file = fold / "predictions.npz"
    np.savez_compressed(
        prediction_file, label_starts=label_starts,
        label_timestamps=np.asarray([str(cities[target]["times"][int(s)]) for s in label_starts]),
        site_ids=np.asarray(cities[target]["node_ids"], dtype=str),
        truth_volume_raw=truth,
        prediction_graphwavenet_fixed_distance_volume_only_raw=prediction,
    )
    state_file = fold / "state.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, state_file)
    row = {
        "experiment_id": "R10-GWN-FIXED-DISTANCE-VOLUME-ONLY-20261031",
        "status": "completed",
        "target_city": target,
        "seed": int(seed),
        "source_cities": list(source_ids),
        "input_variables": ["volume"],
        "target_variable": "volume",
        "metrics_raw_volume": metrics(prediction, truth),
        "validation_label_start_index": TRAIN_END,
        "validation_label_end_exclusive": VAL_END,
        "validation_windows": int(len(label_starts)),
        "site_count": int(len(cities[target]["node_ids"])),
        "protocol": protocol_payload(),
        "training": train_info,
        "architecture": {
            "name": "pinned Graph WaveNet gated temporal backbone with fixed distance diffusion",
            "upstream_commit": UPSTREAM_COMMIT,
            "input_channels": 1,
            "hidden_channels": 32,
            "blocks": 1,
            "layers_per_block": 8,
            "receptive_field": int(model.network.receptive_field),
            "dropout": 0.3,
            "static_distance_graph_top_k": TOP_K,
            "adaptive_node_embedding": False,
            "addaptadj": False,
            "claim_label": "fixed-distance Graph WaveNet",
        },
        "scaler": {"mean_volume": mean, "std_volume": std, "fit_scope": "source cities and train prefix only"},
        "prediction_file": str(prediction_file),
        "state_file": str(state_file),
        "provenance": provenance,
        "code_sha256": source_code_hashes(),
    }
    result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return row


def write_summary(output_root: Path, rows: list[dict[str, Any]], *, status: str) -> None:
    rows = sorted(rows, key=lambda row: (row.get("target_city", ""), int(row.get("seed", 0))))
    summary = {
        "experiment_id": "R10-GWN-FIXED-DISTANCE-VOLUME-ONLY-20261031",
        "status": status,
        "fold_count": len(rows),
        "expected_fold_count": len(CITIES) * len(SEEDS),
        "claim_label": "fixed-distance Graph WaveNet; not adaptive GWN",
        "folds": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target_city", "seed", "rmse", "mae", "validation_windows", "site_count", "status"])
        writer.writeheader()
        for row in rows:
            metric = row.get("metrics_raw_volume", {})
            writer.writerow({"target_city": row.get("target_city"), "seed": row.get("seed"),
                             "rmse": metric.get("rmse"), "mae": metric.get("mae"),
                             "validation_windows": row.get("validation_windows"),
                             "site_count": row.get("site_count"), "status": row.get("status")})


def write_manifest(output_root: Path, rows: list[dict[str, Any]], *, status: str, smoke: dict[str, Any] | None = None) -> None:
    files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json"}:
            files[str(path.relative_to(output_root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "manifest_type": "r10_spatial_baseline_result",
        "manifest_version": 1,
        "experiment_id": "R10-GWN-FIXED-DISTANCE-VOLUME-ONLY-20261031",
        "status": status,
        "write_scope": "experiments/evaluation/r10_spatial_baselines/",
        "claim_label": "fixed-distance Graph WaveNet; not adaptive Graph WaveNet, MTGNN or DCRNN",
        "protocol": protocol_payload(),
        "fold_count": len(rows),
        "expected_fold_count": len(CITIES) * len(SEEDS),
        "completed_folds": [f"{row.get('target_city')}_seed{row.get('seed')}" for row in rows if row.get("status") == "completed"],
        "smoke": smoke,
        "runner_sha256": sha256_file(HERE),
        "source_code_sha256": source_code_hashes(),
        "artifacts": files,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_smoke(output_root: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261031)
    nodes, context, horizon = 11, 16, 4
    coords = rng.normal(size=(nodes, 2))
    distance = np.sqrt(np.square(coords[:, None] - coords[None, :]).sum(axis=-1)).astype(np.float32)
    graph = distance_supports(distance, top_k=3, sigma=1.0)
    model = ChargedFixedGraphWaveNet(channels=1, seq_len=context, pred_len=horizon,
                                     hidden=4, blocks=1, layers=3, dropout=0.0)
    x = torch.randn(2, 1, nodes, context)
    y = torch.randn(2, horizon, nodes)
    prediction = model(x, graph)
    loss = F.mse_loss(prediction, y)
    loss.backward()
    if tuple(prediction.shape) != (2, horizon, nodes) or not torch.isfinite(prediction).all():
        raise RuntimeError(f"unexpected smoke output: {tuple(prediction.shape)}")
    names = [name for name, _ in model.named_parameters()]
    if any("nodevec" in name.lower() for name in names):
        raise RuntimeError("smoke unexpectedly contains learned node embeddings")
    result = {
        "status": "completed",
        "synthetic": True,
        "input_shape": [2, 1, nodes, context],
        "output_shape": list(prediction.shape),
        "finite_output": bool(torch.isfinite(prediction).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "backward_pass": True,
        "adaptive_node_embedding": False,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(output_root, [], status="smoke_completed", smoke=result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true", help="run all 18 city×seed folds")
    parser.add_argument("--target", choices=CITIES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        if args.all or args.target or args.seed:
            parser.error("--smoke cannot be combined with fold selection")
        print(json.dumps(run_smoke(args.output_root), ensure_ascii=False, sort_keys=True))
        return 0
    if args.all:
        if args.target or args.seed:
            parser.error("--all cannot be combined with --target/--seed")
        args.output_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for target in CITIES:
            for seed in SEEDS:
                rows.append(run_fold(target=target, seed=seed, output_root=args.output_root,
                                     steps=args.steps, device=args.device,
                                     torch_threads=args.torch_threads, force=args.force))
                write_summary(args.output_root, rows, status="running")
                write_manifest(args.output_root, rows, status="running")
        write_summary(args.output_root, rows, status="completed")
        write_manifest(args.output_root, rows, status="completed")
        print(json.dumps({"status": "completed", "fold_count": len(rows)}, ensure_ascii=False))
        return 0
    if not args.target or args.seed is None:
        parser.error("provide --smoke, --all, or both --target and --seed")
    row = run_fold(target=args.target, seed=args.seed, output_root=args.output_root,
                   steps=args.steps, device=args.device, torch_threads=args.torch_threads,
                   force=args.force)
    rows = [row]
    write_summary(args.output_root, rows, status="fold_completed")
    write_manifest(args.output_root, rows, status="fold_completed")
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

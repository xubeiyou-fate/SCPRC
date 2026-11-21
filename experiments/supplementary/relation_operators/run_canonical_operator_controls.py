#!/usr/bin/env python3
"""Run Generic Attention and DeepSets controls in an isolated canonical output tree."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EXPERIMENTS = ROOT / 'scprc/training'
PROTOCOL_PATH = EXPERIMENTS / 'scprc_matched_protocol.json'
EXPECTED_PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
ATTENTION_RUNNER = EXPERIMENTS / "run_charged_relation_controls.py"
DEEPSETS_RUNNER = ROOT / "experiments/operators/deepsets_adapter_loco/run_deepsets_adapter_loco.py"
MODEL_PATHS = {
    "generic_attention": ROOT / "scprc/models/phasewise_attention_set_adapter.py",
    "deepsets": ROOT / "scprc/models/phaseformer_deepsets_adapter.py",
}
TARGETS = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
DEEPSETS_WORKER_CONTEXT: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_record() -> dict[str, Any]:
    digest = sha256(PROTOCOL_PATH)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"canonical protocol hash mismatch: {digest}")
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if tuple(value["data_split"]["cities"]) != TARGETS:
        raise RuntimeError("frozen city list mismatch")
    if tuple(value["training_budget"]["seeds"]) != SEEDS:
        raise RuntimeError("frozen seed list mismatch")
    return {"protocol": value, "protocol_sha256": digest}


def initialize_deepsets_worker(device: str, torch_threads: int) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXPERIMENTS))
    sys.path.insert(0, str(DEEPSETS_RUNNER.parent))
    import scprc.training.training_protocol as protocol
    import experiments.operators.deepsets_adapter_loco.run_deepsets_adapter_loco as deepsets

    cities, provenance = protocol.load_cities()
    deepsets.RUNS = OUT / "deepsets" / "runs"
    deepsets.OUT = OUT / "deepsets"
    deepsets.RUNS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch_threads)
    DEEPSETS_WORKER_CONTEXT.update({
        "cities": cities,
        "provenance": provenance,
        "module": deepsets,
        "device": device,
        "torch_threads": torch_threads,
    })


def run_deepsets_fold_task(task: tuple[str, int]) -> dict[str, Any]:
    target, seed = task
    fold = OUT / "deepsets" / "runs" / f"target_{target.lower()}_seed{seed}" / "attempt_01"
    result_path = fold / "result.json"
    prediction_path = fold / "predictions.npz"
    checkpoint_path = fold / "trained_checkpoint.pt"
    if all(path.is_file() for path in (result_path, prediction_path, checkpoint_path)):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return {"target": target, "seed": seed, "metrics": cached["metrics"],
                    "result": str(result_path), "resumed": True}
    if fold.exists():
        raise RuntimeError(f"incomplete DeepSets fold directory already exists: {fold}")
    result = DEEPSETS_WORKER_CONTEXT["module"].run_fold(
        target, seed, DEEPSETS_WORKER_CONTEXT["cities"],
        DEEPSETS_WORKER_CONTEXT["provenance"],
        device=DEEPSETS_WORKER_CONTEXT["device"],
        torch_threads=DEEPSETS_WORKER_CONTEXT["torch_threads"],
        attempt=1,
    )
    return {"target": target, "seed": seed, "metrics": result["metrics"],
            "result": str(result_path), "resumed": False}


def worker(model_key: str, device: str, torch_threads: int) -> dict[str, Any]:
    frozen = protocol_record()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXPERIMENTS))
    sys.path.insert(0, str(DEEPSETS_RUNNER.parent))
    completed = []
    if model_key == "generic_attention":
        import scprc.training.training_protocol as protocol
        import scprc.training.run_charged_relation_controls as attention

        cities, provenance = protocol.load_cities()
        model_root = OUT / "generic_attention"
        for target in TARGETS:
            for seed in SEEDS:
                result = attention.run_fold(
                    "PhaseWiseAttention_relation", target, seed,
                    cities=cities, data_provenance=provenance,
                    output_root=model_root, steps=2048, device=device,
                    torch_threads=torch_threads, overwrite=False,
                )
                completed.append({"target": target, "seed": seed,
                                  "metrics": result["metrics"],
                                  "result": str(model_root / f"target_{target.lower()}_seed{seed}" / "result.json")})
    elif model_key == "deepsets":
        tasks = [(target, seed) for target in TARGETS for seed in SEEDS]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_deepsets_worker,
            initargs=(device, torch_threads),
        ) as pool:
            futures = [pool.submit(run_deepsets_fold_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
    else:
        raise ValueError(model_key)
    return {"model": model_key, "completed_folds": completed}


def collect_matrix(model_key: str) -> dict[str, Any]:
    model_root = OUT / model_key
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            fold = (model_root / f"target_{target.lower()}_seed{seed}"
                    if model_key == "generic_attention" else
                    model_root / "runs" / f"target_{target.lower()}_seed{seed}" / "attempt_01")
            result_path = fold / "result.json"
            prediction_path = fold / "predictions.npz"
            checkpoint_path = fold / "trained_checkpoint.pt"
            if not all(path.is_file() for path in (result_path, prediction_path, checkpoint_path)):
                raise FileNotFoundError(f"incomplete fold artifacts: {fold}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            protocol_meta = result.get("protocol", {})
            if protocol_meta.get("protocol_id") != "scpcr-matched-baseline-20261023":
                raise RuntimeError(f"protocol ID mismatch: {result_path}")
            if protocol_meta.get("protocol_sha256", protocol_meta.get("frozen_protocol_sha256")) != EXPECTED_PROTOCOL_SHA256:
                raise RuntimeError(f"protocol hash mismatch: {result_path}")
            metrics = result["metrics"]
            if set(("rmse", "mae")) - set(metrics):
                raise RuntimeError(f"missing primary metrics: {result_path}")
            with np.load(prediction_path, allow_pickle=False) as archive:
                starts_key = "label_starts"
                if starts_key not in archive or len(archive[starts_key]) != 721:
                    raise RuntimeError(f"evaluation-origin mismatch: {prediction_path}")
            rows.append({
                "target": target,
                "seed": seed,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "result": str(result_path.relative_to(OUT)),
                "predictions": str(prediction_path.relative_to(OUT)),
                "checkpoint": str(checkpoint_path.relative_to(OUT)),
                "result_sha256": sha256(result_path),
                "predictions_sha256": sha256(prediction_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "training": result["training"],
            })
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 folds for {model_key}, got {len(rows)}")
    return {
        "status": "completed",
        "model": model_key,
        "fold_count": len(rows),
        "equal_fold_mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "equal_fold_mean_mae": float(np.mean([row["mae"] for row in rows])),
        "folds": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    existing_manifest = OUT / "manifest.json"
    previous_manifest = None
    previous_manifest_sha256 = None
    if existing_manifest.is_file():
        if not args.resume:
            parser.error(f"run manifest exists; pass --resume to continue: {existing_manifest}")
        previous_manifest_sha256 = sha256(existing_manifest)
        previous_manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if previous_manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            parser.error("cannot resume: existing manifest uses a different frozen protocol")
        if previous_manifest.get("status") not in ("running", "incomplete"):
            parser.error(f"cannot resume a run with status {previous_manifest.get('status')!r}")
    elif args.resume:
        parser.error(f"cannot resume without an existing run manifest: {existing_manifest}")
    if not args.resume and any((OUT / name).exists() for name in ("generic_attention", "deepsets")):
        parser.error(f"model output paths already exist; refusing to overwrite: {OUT}")

    frozen = protocol_record()
    write_json(OUT / "protocol.json", frozen["protocol"])
    (OUT / "command.txt").write_text(
        f"python {Path(__file__).resolve()} --device {args.device} --torch-threads {args.torch_threads} --parallel-workers {args.parallel_workers} --resume\n",
        encoding="utf-8",
    )
    write_json(OUT / "environment.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": args.device,
        "torch_threads_per_worker": args.torch_threads,
        "parallel_workers": args.parallel_workers,
        "execution_note": "City-seed folds run in isolated spawn-based processes with fold seeds reset inside each runner; no model state or RNG is shared between workers.",
        "protocol_sha256": frozen["protocol_sha256"],
    })
    code_files = [Path(__file__).resolve(), ATTENTION_RUNNER, DEEPSETS_RUNNER,
                  PROTOCOL_PATH, *MODEL_PATHS.values(),
                  ROOT / "scprc/models/phasewise_attention_set_adapter.py",
                  ROOT / "scprc/training/nbeats_graph.py",
                  EXPERIMENTS / "training_protocol.py"]
    manifest = {
        "status": "running",
        "protocol_id": frozen["protocol"]["protocol_id"],
        "protocol_sha256": frozen["protocol_sha256"],
        "models": ["generic_attention", "deepsets"],
        "targets": list(TARGETS), "seeds": list(SEEDS),
        "expected_folds_per_model": 18,
        "expected_total_folds": 36,
        "budget": {
            "source_minibatches": 2048,
            "batch_size_windows": 4,
            "effective_source_windows": 8192,
            "pcgrad_rounds": 410,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": frozen["protocol"]["training_budget"]["loss"],
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                           for path in code_files if path.is_file()},
        "started_at_unix": previous_manifest.get("started_at_unix", time.time()) if previous_manifest else time.time(),
        "resumed_at_unix": time.time() if previous_manifest else None,
        "resume_parent_manifest_sha256": previous_manifest_sha256,
    }
    write_json(OUT / "manifest.json", manifest)

    worker.parallel_workers = args.parallel_workers
    failures = []
    for model_name in ("generic_attention", "deepsets"):
        try:
            worker(model_name, args.device, args.torch_threads)
        except Exception as error:
            failures.append({"model": model_name, "error": repr(error)})
            write_json(OUT / f"{model_name}_failure.json", failures[-1])

    matrices = {}
    for model_name in ("generic_attention", "deepsets"):
        try:
            matrices[model_name] = collect_matrix(model_name)
        except Exception as error:
            failures.append({"model": model_name, "audit_error": repr(error)})
    complete = not failures and all(
        matrices.get(name, {}).get("fold_count") == 18
        for name in ("generic_attention", "deepsets")
    )
    summary = {
        "status": "completed" if complete else "incomplete",
        "protocol_id": frozen["protocol"]["protocol_id"],
        "protocol_sha256": frozen["protocol_sha256"],
        "expected_total_folds": 36,
        "completed_total_folds": sum(matrix.get("fold_count", 0) for matrix in matrices.values()),
        "matrices": matrices,
        "failures": failures,
        "interpretation": "Retrospective August controls; metrics are descriptive equal-fold means. No Online/support/fallback post-processing is applied.",
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "manifest.json", {
        **manifest,
        "status": summary["status"],
        "completed_total_folds": summary["completed_total_folds"],
        "finished_at_unix": time.time(),
        "failures": failures,
    })
    lines = [
        "# Canonical relation operator controls (2026-10-27)",
        "",
        f"- Status: **{summary['status']}**",
        f"- Protocol: `{frozen['protocol']['protocol_id']}` (SHA-256 `{frozen['protocol_sha256']}`)",
        f"- Completed: {summary['completed_total_folds']}/36 folds",
        "- August is retrospective; candidate forecasts are scored directly without Online, support projection, or Exact fallback.",
        "",
        "| Operator control | Folds | Equal-fold RMSE | Equal-fold MAE |",
        "|---|---:|---:|---:|",
    ]
    for name in ("generic_attention", "deepsets"):
        matrix = matrices.get(name)
        if matrix:
            lines.append(f"| {name} | {matrix['fold_count']}/18 | {matrix['equal_fold_mean_rmse']:.6f} | {matrix['equal_fold_mean_mae']:.6f} |")
        else:
            lines.append(f"| {name} | incomplete | — | — |")
    lines.extend(["", "Full fold metrics, artifact hashes, and training records are in `summary.json`; frozen settings and code hashes are in `manifest.json`.", ""])
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

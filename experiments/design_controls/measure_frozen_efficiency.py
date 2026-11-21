#!/usr/bin/env python3
"""Measure frozen forward efficiency; never train, resume, or fit a scaler.

No-argument paths are relative to this file, not the working directory. CUDA
measurement is intended for AFTER training has finished and the GPU is idle.
An incomplete input snapshot is exported without importing torch unless
--allow-partial is explicitly supplied. --help uses only the standard library.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
VARIANTS = ("full_same_run", "phase_shared_strict",
            "stationwise_matched", "concat_project")
BATCHES = (1, 4, 16)
N_ORIGINS, VALIDATE_ORIGINS = 16, 4
RTOL, ATOL = 1e-5, 1e-3
TIMING_SCOPE = "pure_forward_data_transfer_excluded"

CASE_FIELDS = (
    "city", "seed", "variant", "batch_size", "status", "detail",
    "device", "precision", "timing_scope", "n_stations", "available_origins",
    "selected_origins", "label_starts", "batches_per_rep", "warmup", "reps",
    "parameters_total", "parameters_trainable_registered",
    "parameters_backbone", "parameters_relation", "parameters_fusion",
    "validation_passed", "validation_rtol", "validation_atol",
    "validation_max_abs", "validation_max_relative_error",
    "validation_max_rtol_deviation", "validation_zero_reference_mismatches",
    "validation_violation_count", "validation_relative_only_violation_count",
    "validation_max_tolerance_excess", "validation_elements",
    "wall_ms_per_batch_mean", "wall_ms_per_batch_median",
    "wall_ms_per_batch_std", "wall_ms_per_origin_mean", "wall_origins_per_second",
    "gpu_ms_per_batch_mean", "gpu_ms_per_batch_median",
    "gpu_ms_per_batch_std", "gpu_ms_per_origin_mean", "gpu_origins_per_second",
    "cuda_baseline_allocated_bytes", "cuda_peak_allocated_bytes",
    "cuda_incremental_peak_allocated_bytes", "cuda_baseline_reserved_bytes",
    "cuda_peak_reserved_bytes", "cuda_incremental_peak_reserved_bytes",
    "wall_sweep_ms", "gpu_sweep_ms", "checkpoint_sha256",
    "prediction_sha256", "fold_protocol_sha256", "fold_data_sha256",
    "values_sha256", "normalized_histories_sha256",
)
MEAN_FIELDS = (
    "parameters_total", "parameters_trainable_registered",
    "parameters_backbone", "parameters_relation", "parameters_fusion",
    "wall_ms_per_batch_mean", "wall_ms_per_origin_mean", "wall_origins_per_second",
    "gpu_ms_per_batch_mean", "gpu_ms_per_origin_mean", "gpu_origins_per_second",
    "cuda_baseline_allocated_bytes", "cuda_peak_allocated_bytes",
    "cuda_incremental_peak_allocated_bytes", "cuda_baseline_reserved_bytes",
    "cuda_peak_reserved_bytes", "cuda_incremental_peak_reserved_bytes",
)
SUMMARY_FIELDS = ("variant", "batch_size", "device", "precision", "timing_scope", "n_folds",
                  "expected_folds", "n_cities", "complete", "estimator",
                  "status_counts") + MEAN_FIELDS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def atomic_text(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_text(path, lambda stream: json.dump(
        data, stream, indent=2, sort_keys=True, allow_nan=False))


def write_csv(path, fields, rows):
    def emit(stream):
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    atomic_text(path, emit)


class Fingerprints:
    """Hash once per unchanged file, retaining a snapshot consistency check."""

    def __init__(self):
        self.files = {}

    @staticmethod
    def signature(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def record(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        key = str(path)
        before = self.signature(path)
        if key in self.files:
            require(tuple(self.files[key]["signature"]) == before,
                    f"input changed during snapshot: {path}")
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            require(before == self.signature(path), f"input changed while hashing: {path}")
            self.files[key] = {"sha256": digest.hexdigest(), "bytes": before[2],
                               "signature": list(before)}
        value = self.files[key]["sha256"]
        if expected is not None:
            require(value == expected, f"SHA-256 mismatch: {path}")
        return value

    def check_unchanged(self):
        changed = []
        for path, record in self.files.items():
            try:
                if self.signature(Path(path)) != tuple(record["signature"]):
                    changed.append(path)
            except OSError:
                changed.append(path)
        return changed


def inventory(root, fingerprints):
    """Bounded 72-run discovery; smoke/benchmarks cannot fill a missing run."""
    entries = []
    for city in CITIES:
        for seed in SEEDS:
            fold = root / "runs" / f"target_{city.lower()}_seed{seed}"
            for variant in VARIANTS:
                directory = fold / variant
                entry = dict(city=city, seed=seed, variant=variant, fold=fold,
                             directory=directory, status="ready", detail="")
                required = (directory / "result.json", directory / "checkpoint.pt",
                            directory / "prediction_raw.npy", fold / "fold_protocol.json",
                            fold / "fold_data.npz", root / "data_cache" / f"{city}_values.npy",
                            root / "data_cache" / "metadata.json")
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    entry.update(status="missing_artifacts", detail="; ".join(missing))
                else:
                    try:
                        fingerprints.record(required[0])
                        result = read_json(required[0])
                        require(isinstance(result, dict), "result must be a JSON object")
                        require(result.get("status") == "completed",
                                "result is not completed full-budget training")
                        for key, expected in (("target_city", city), ("seed", seed),
                                              ("variant", variant), ("steps", 2048),
                                              ("window_exposures", 8192), ("optimizer_steps", 410)):
                            require(result.get(key) == expected, f"result {key} != {expected}")
                        require(result.get("parameter_count", 0) > 0, "missing parameter count")
                        require(isinstance(result.get("environment"), dict),
                                "missing training environment")
                        entry["result"] = result
                    except (ValueError, OSError, TypeError) as exc:
                        entry.update(status="invalid_artifacts", detail=str(exc))
                entries.append(entry)
    return entries


def load_histories(entry, root, fingerprints, np):
    fold, result, city = entry["fold"], entry["result"], entry["city"]
    cfg_path, data_path = fold / "fold_protocol.json", fold / "fold_data.npz"
    cfg_hash = fingerprints.record(cfg_path)
    cfg = read_json(cfg_path)
    require(cfg == result.get("fold_protocol"), "result/fold protocol mismatch")
    for key, expected in (("target_city", city), ("seed", entry["seed"]),
                          ("context", 168), ("horizon", 24),
                          ("evaluation_label_starts", [2928, 3649])):
        require(cfg.get(key) == expected, f"fold protocol {key} mismatch")
    require(set(cfg.get("source_cities", [])) == set(CITIES) - {city},
            "source city split mismatch")
    require(cfg.get("support_scale", 1.0) == 1.0
            and cfg.get("target_support_projection") is False
            and cfg.get("target_release_gate") is False, "unsupported target adaptation")
    mean, std = (float(cfg["source_scaler"][key]) for key in ("mean", "std"))
    require(np.isfinite([mean, std]).all() and std > 0, "invalid saved source scaler")
    require(np.isfinite(cfg["source_relation_threshold"])
            and cfg["source_relation_threshold"] > 0, "invalid saved source threshold")
    data_hash = fingerprints.record(data_path, cfg.get("fold_data_sha256"))
    # Read origin/site metadata only. No future truth array is accessed.
    with np.load(data_path, allow_pickle=False) as data:
        starts = data["label_starts"]
        sites = data["site_ids"].astype(str).tolist()
    require(starts.ndim == 1 and starts.dtype.kind in "iu", "invalid label_starts")
    require(len(starts) <= 721 and np.array_equal(starts, np.arange(2928, 2928 + len(starts))),
            "origins are not the ordered evaluation prefix")
    require(sites and len(set(sites)) == len(sites), "invalid station identifiers")
    cache = root / "data_cache"
    fingerprints.record(cache / "metadata.json")
    meta = read_json(cache / "metadata.json")["cities"][city]
    require(sites == [str(site) for site in meta["site_ids"]], "station order mismatch")
    values_path = cache / f"{city}_values.npy"
    values_hash = fingerprints.record(values_path, meta.get("values_sha256"))
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    require(values.ndim == 2 and values.dtype == np.float32
            and values.shape[1] == len(sites), "expected native FP32 [time,stations] values")
    selected = starts[:N_ORIGINS]
    require(len(selected) == 0 or (selected[0] >= 168 and selected[-1] <= len(values)),
            "insufficient input history rows")
    info = dict(n_stations=len(sites), available_origins=len(starts),
                selected_origins=len(selected), label_starts=selected.tolist(),
                fold_protocol_sha256=cfg_hash, fold_data_sha256=data_hash, values_sha256=values_hash)
    if len(selected) < N_ORIGINS:
        return cfg, None, info
    # Exactly the native runner's FP32 conversion and NumPy scalar arithmetic.
    history = np.stack([values[s - 168:s] for s in selected]).astype(np.float32)
    history = (history - mean) / std
    require(history.dtype == np.float32 and np.isfinite(history).all(), "nonfinite/invalid histories")
    info["normalized_histories_sha256"] = hashlib.sha256(history.tobytes(order="C")).hexdigest()
    return cfg, np.ascontiguousarray(history), info


def validation_metrics(actual, reference, np):
    require(actual.shape == reference.shape and actual.dtype == np.float32
            and reference.dtype == np.float32, "native prediction shape/dtype mismatch")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(),
            "nonfinite native validation predictions")
    diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denom = np.abs(reference.astype(np.float64))
    nonzero = denom > 0
    relative = diff[nonzero] / denom[nonzero]
    adjusted = np.maximum(diff[nonzero] - ATOL, 0) / denom[nonzero]
    violation = diff > (ATOL + RTOL * denom)
    return {
        "validation_passed": not bool(violation.any()),
        "validation_rtol": RTOL, "validation_atol": ATOL,
        "validation_max_abs": float(diff.max()),
        "validation_max_relative_error": float(relative.max()) if relative.size else None,
        "validation_max_rtol_deviation": float(np.maximum(adjusted - RTOL, 0).max())
        if adjusted.size else None,
        "validation_zero_reference_mismatches": int(((~nonzero) & (diff > 0)).sum()),
        "validation_violation_count": int(violation.sum()),
        "validation_relative_only_violation_count": int((diff > RTOL * denom).sum()),
        "validation_max_tolerance_excess": float(np.maximum(diff - ATOL - RTOL * denom, 0).max()),
        "validation_elements": int(diff.size),
    }


def parameter_counts(model):
    count = lambda module: sum(p.numel() for p in module.parameters()) if module is not None else 0
    return dict(parameters_total=count(model),
                parameters_trainable_registered=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameters_backbone=count(model.backbone), parameters_relation=count(model.relation),
                parameters_fusion=count(getattr(model, "fusion", None)))


def measure_case(model, histories, batch_size, warmup, reps, torch, np):
    """One repetition sweeps all 16 fixed histories in contiguous batches."""
    device = histories.device
    cuda = device.type == "cuda"
    batches = tuple(histories.split(batch_size))
    require(len(batches) * batch_size == N_ORIGINS, "no padding or repeated histories allowed")
    if cuda:
        # Initialize lazy event resources outside both warmup and measurement.
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        end.record()
        torch.cuda.synchronize(device)
    wall, gpu = [], []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for _ in range(warmup):
            for batch in batches:
                output = model(batch)
                del output
        if cuda:
            torch.cuda.synchronize(device)
            memory = {"cuda_baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                      "cuda_baseline_reserved_bytes": int(torch.cuda.memory_reserved(device))}
            torch.cuda.reset_peak_memory_stats(device)
        else:
            memory = {}
        for _ in range(reps):
            sweep_wall = sweep_gpu = 0.0
            for batch in batches:
                if cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                if cuda:
                    begin.record()
                output = model(batch)
                if cuda:
                    end.record()
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                sweep_wall += elapsed
                if cuda:
                    sweep_gpu += begin.elapsed_time(end)
                # Checks, copies, and frees are outside the forward clock.
                require(output.dtype == torch.float32 and tuple(output.shape)
                        == (batch_size, 24, histories.shape[2]), "invalid timed output")
                require(bool(torch.isfinite(output).all()), "nonfinite timed output")
                del output
            wall.append(sweep_wall)
            if cuda:
                gpu.append(sweep_gpu)
        if cuda:
            memory.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                          cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            for kind in ("allocated", "reserved"):
                memory[f"cuda_incremental_peak_{kind}_bytes"] = (
                    memory[f"cuda_peak_{kind}_bytes"] - memory[f"cuda_baseline_{kind}_bytes"])
    stats = dict(batches_per_rep=len(batches), wall_sweep_ms=wall, gpu_sweep_ms=gpu or None, **memory)
    for label, samples in (("wall", wall), ("gpu", gpu)):
        if samples:
            per_batch = [sample / len(batches) for sample in samples]
            average = statistics.mean(samples)
            stats.update({f"{label}_ms_per_batch_mean": statistics.mean(per_batch),
                          f"{label}_ms_per_batch_median": statistics.median(per_batch),
                          f"{label}_ms_per_batch_std": statistics.stdev(per_batch) if reps > 1 else 0.0,
                          f"{label}_ms_per_origin_mean": average / N_ORIGINS,
                          f"{label}_origins_per_second": 1000 * N_ORIGINS / average})
    return stats


def environment(torch, np, device):
    result = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                  cuda=torch.version.cuda, device=str(device), platform=platform.platform(),
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
                  omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                  mkl_num_threads=os.environ.get("MKL_NUM_THREADS"))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(device=torch.cuda.get_device_name(device), cuda_index=device.index,
                      gpu_total_memory_bytes=props.total_memory,
                      gpu_compute_capability=[props.major, props.minor], cudnn=torch.backends.cudnn.version())
    return result


def summarize(rows, device):
    output = []
    for variant in VARIANTS:
        for batch in BATCHES:
            cases = [row for row in rows if row["variant"] == variant and row["batch_size"] == batch]
            good = [row for row in cases if row["status"] == "measured"]
            counts = {}
            for row in cases:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            summary = dict(variant=variant, batch_size=batch, device=device, precision="float32_no_AMP",
                           timing_scope=TIMING_SCOPE,
                           n_folds=len(good), expected_folds=18, n_cities=len({r["city"] for r in good}),
                           complete=len(good) == 18, status_counts=counts,
                           estimator="equal_18_fold_mean" if len(good) == 18
                           else "equal_available_fold_mean_monitoring")
            for field in MEAN_FIELDS:
                values = [row[field] for row in good if row.get(field) is not None]
                summary[field] = statistics.mean(values) if len(values) == len(good) and values else None
            output.append(summary)
    return output


def run(args):
    root = args.root.resolve()
    out = (args.out or root / "efficiency").resolve()
    # Outputs cannot be redirected over any experiment inputs.
    for protected in (root / "runs", root / "data_cache", HERE / "code_snapshots"):
        require(out != protected and protected not in out.parents, "output overlaps protected inputs")
    fingerprints = Fingerprints()
    source_paths = [Path(__file__), HERE / "model_variants.py", HERE / "METHODS_AND_LIMITS.md"]
    for path in source_paths:
        fingerprints.record(path)
    entries = inventory(root, fingerprints)
    ready = sum(entry["status"] == "ready" for entry in entries)
    rows = []
    manifest = dict(schema_version=1, status="running", complete=False,
                    started_at_utc=datetime.now(timezone.utc).isoformat(), root=str(root), out=str(out),
                    expected_runs=72, expected_folds_per_variant=18, expected_cases=216, ready_runs=ready,
                    settings=dict(device=args.device, warmup=args.warmup, reps=args.reps,
                                  torch_threads=args.torch_threads, allow_partial=args.allow_partial,
                                  batch_sizes=list(BATCHES), histories=N_ORIGINS, validation_histories=4,
                                  rtol=RTOL, atol=ATOL, precision="float32_no_AMP"),
                    selection="first 16 evaluation label_starts, fixed prefix; no outcome selection",
                    timing_scope=TIMING_SCOPE,
                    timing="pure forward, data transfer excluded; each sweep uses all 16 histories; wall and CUDA events separate",
                    training_invoked=False, cuda_initialized=False, run_records=[])
    write_json(out / "manifest.json", manifest)
    runtime = None
    torch = np = build_model = device = None
    should_measure = ready == 72 or (args.allow_partial and ready > 0)
    if should_measure:
        try:
            import numpy as np
            import torch
            sys.path.insert(0, str(PROJECT))
            from experiments.design_controls.model_variants import build_model
            torch.set_num_threads(args.torch_threads)
            device = torch.device(args.device)
            require(device.type in ("cpu", "cuda"), "only CPU/CUDA supported")
            # Match recorded numerical policy across every measured model.
            training_env = next(entry["result"]["environment"] for entry in entries if entry["status"] == "ready")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = bool(training_env.get("cudnn_deterministic", False))
            torch.use_deterministic_algorithms(bool(training_env.get("deterministic_algorithms", False)))
            if device.type == "cuda":
                require(torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
                torch.cuda.set_device(device)
                device = torch.device("cuda", torch.cuda.current_device())
                manifest["cuda_initialized"] = True
            runtime = environment(torch, np, device)
            manifest["environment"] = runtime
        except Exception as exc:
            manifest["setup_error"] = repr(exc)
            should_measure = False
    for entry in entries:
        base = dict(city=entry["city"], seed=entry["seed"], variant=entry["variant"],
                    status=entry["status"], detail=entry["detail"], device=args.device,
                    precision="float32_no_AMP", timing_scope=TIMING_SCOPE, warmup=args.warmup, reps=args.reps,
                    validation_rtol=RTOL, validation_atol=ATOL)
        record = {key: base[key] for key in ("city", "seed", "variant", "status", "detail")}
        case_rows = []
        model = histories = checkpoint = state = None
        try:
            if base["status"] != "ready":
                pass
            elif not should_measure:
                base.update(status="blocked_incomplete_snapshot" if ready < 72 and not args.allow_partial
                            else "setup_failed", detail=manifest.get("setup_error",
                            "all 72 full-budget runs required; no CUDA work attempted"))
            else:
                result = entry["result"]
                cfg, history, info = load_histories(entry, root, fingerprints, np)
                base.update(info)
                record.update(training_environment=result["environment"],
                              training_source_hashes=result.get("source_hashes", {}))
                if history is None:
                    base.update(status="insufficient_cases", detail="fewer than 16 fixed evaluation origins; no padding/resampling")
                else:
                    if device.type == "cuda":
                        for key in ("python", "torch", "numpy", "cuda", "device", "tf32_matmul",
                                    "cudnn_deterministic", "deterministic_algorithms"):
                            require(key in result["environment"] and result["environment"][key] == runtime[key],
                                    f"training/measurement environment mismatch: {key}")
                    directory = entry["directory"]
                    base["checkpoint_sha256"] = fingerprints.record(directory / "checkpoint.pt", result.get("checkpoint_sha256"))
                    base["prediction_sha256"] = fingerprints.record(directory / "prediction_raw.npy", result.get("predictions_sha256"))
                    # These local runner checkpoints include NumPy RNG objects. Only
                    # trusted local artifacts may be unpickled. Optimizer state is unused.
                    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
                    checkpoint_protocol = checkpoint.get("fold_protocol")
                    if isinstance(checkpoint_protocol, dict) and isinstance(checkpoint_protocol.get("source_cities"), tuple):
                        checkpoint_protocol = {**checkpoint_protocol, "source_cities": list(checkpoint_protocol["source_cities"])}
                    require(checkpoint.get("variant") == entry["variant"]
                            and checkpoint_protocol == cfg, "checkpoint identity/protocol mismatch")
                    require(checkpoint.get("source_hashes", {}) == result.get("source_hashes", {}),
                            "checkpoint/result source hash mismatch")
                    for relative, expected in result.get("source_hashes", {}).items():
                        path = (PROJECT / relative).resolve()
                        require(PROJECT in path.parents, "source path outside project")
                        fingerprints.record(path, expected)
                    state = checkpoint["state_dict"]
                    require(all(not value.is_floating_point() or value.dtype == torch.float32
                                for value in state.values()), "checkpoint is not FP32")
                    model = build_model(entry["variant"], info["n_stations"],
                                        cfg["source_relation_threshold"], seed=entry["seed"])
                    model.load_state_dict(state, strict=True)
                    model.float().to(device).eval()
                    base.update(parameter_counts(model))
                    require(base["parameters_trainable_registered"] == result["parameter_count"],
                            "actual/recorded parameter count mismatch")
                    del checkpoint, state
                    checkpoint = state = None
                    histories = torch.from_numpy(history).to(device=device, dtype=torch.float32)
                    prediction = np.load(directory / "prediction_raw.npy", mmap_mode="r", allow_pickle=False)
                    require(prediction.ndim == 3 and prediction.shape == (info["available_origins"], 24, info["n_stations"]),
                            "saved prediction alignment mismatch")
                    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                        native = model(histories[:VALIDATE_ORIGINS]).cpu().numpy()
                    native = (native * cfg["source_scaler"]["std"] + cfg["source_scaler"]["mean"]).astype(np.float32)
                    base.update(validation_metrics(native, prediction[:VALIDATE_ORIGINS], np))
                    del prediction, native
                    if not base["validation_passed"]:
                        base.update(status="validation_failed", detail="native first-four predictions exceed declared tolerances; no retraining or timing")
                    else:
                        for batch in BATCHES:
                            row = dict(base, batch_size=batch)
                            try:
                                row.update(measure_case(model, histories, batch, args.warmup, args.reps, torch, np))
                                row.update(status="measured", detail="")
                            except Exception as exc:
                                row.update(status="measurement_failed", detail=repr(exc))
                            case_rows.append(row)
        except Exception as exc:
            base.update(status="invalid_artifacts", detail=repr(exc))
            record["traceback"] = traceback.format_exc()
        finally:
            del model, histories, checkpoint, state
            if should_measure and device is not None and device.type == "cuda":
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        if not case_rows:
            case_rows = [dict(base, batch_size=batch) for batch in BATCHES]
        rows.extend(case_rows)
        record.update(statuses={str(row["batch_size"]): row["status"] for row in case_rows},
                      status=case_rows[0]["status"] if len({row["status"] for row in case_rows}) == 1 else "mixed",
                      detail=base["detail"])
        manifest["run_records"].append(record)
        if any(row["status"] == "measured" for row in case_rows):
            print(json.dumps({"city": base["city"], "seed": base["seed"],
                              "variant": base["variant"], "statuses": record["statuses"]}), flush=True)
    # Some dynamically generated torch modules have synthetic __file__ names.
    # Only existing project files are provenance inputs.
    source_errors = []
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(path).endswith(".py"):
            path = Path(path).resolve()
            if PROJECT in path.parents and path.is_file():
                try:
                    fingerprints.record(path)
                except (ValueError, OSError):
                    source_errors.append(str(path))
    changed = sorted(set(fingerprints.check_unchanged() + source_errors))
    if changed:
        manifest["changed_inputs"] = changed
        for row in rows:
            if row["status"] == "measured":
                row.update(status="invalidated_changed_snapshot", detail="hashed inputs changed during measurement")
    count = sum(row["status"] == "measured" for row in rows)
    require(len(rows) == 216, "internal case-count error")
    manifest.update(complete=count == 216, measured_cases=count, insufficient_cases=216 - count,
                    status="complete" if count == 216 else "incomplete",
                    finished_at_utc=datetime.now(timezone.utc).isoformat(), inputs=fingerprints.files,
                    missing_or_failed_cases=[{k: row[k] for k in ("city", "seed", "variant", "batch_size", "status", "detail")}
                                             for row in rows if row["status"] != "measured"])
    write_csv(out / "per_fold_efficiency.csv", CASE_FIELDS, rows)
    write_csv(out / "summary.csv", SUMMARY_FIELDS, summarize(rows, args.device))
    manifest["outputs"] = {name: fingerprints.record(out / name)
                           for name in ("per_fold_efficiency.csv", "summary.csv")}
    # Keep output hashes separate from the immutable input inventory.
    output_paths = {str((out / name).resolve()) for name in manifest["outputs"]}
    manifest["inputs"] = {key: value for key, value in fingerprints.files.items()
                          if key not in output_paths}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"complete": manifest["complete"], "measured_cases": count,
                      "expected_cases": 216, "out": str(out)}), flush=True)
    return 0 if count == 216 else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE, help="experiment root containing runs/ and data_cache/ (default: HERE)")
    parser.add_argument("--out", type=Path, help="default: ROOT/efficiency")
    parser.add_argument("--device", default="cuda", help="cuda[:index] for formal measurements; cpu for explicit CPU checks")
    parser.add_argument("--warmup", type=int, default=3, help="untimed sweeps of all 16 histories per batch size (default: 3)")
    parser.add_argument("--reps", type=int, default=10, help="timed sweeps of all 16 histories per batch size (default: 10)")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true", help="measure available completed runs; still incomplete and exit 2 unless all 216 cases succeed")
    args = parser.parse_args(argv)
    if args.device == "cuda":
        args.device = "cuda:0"
    if args.warmup < 0 or args.reps < 1 or args.torch_threads < 1:
        parser.error("warmup >= 0, reps >= 1 and torch-threads >= 1 are required")
    if args.device != "cpu" and args.device != "cuda" and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu, cuda, or cuda:<index>")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

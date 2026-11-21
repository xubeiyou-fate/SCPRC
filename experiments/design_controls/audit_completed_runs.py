#!/usr/bin/env python3
"""Independent CPU artifact audit for the 2026-11-17 design controls.

Reads only saved tensors/arrays/metadata; never imports a training model or runs
training. Default output is ROOT/audit/completed_runs_audit.json. Exit 0 means
all 72 identities passed, exit 1 means incomplete/invalid. A result schema map
can name dotted metadata paths without weakening the checks (see --help).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
PARAMETERS = {"full_same_run": 37263, "phase_shared_strict": 37263,
              "stationwise_matched": 37267, "concat_project": 37368}
VARIANTS = tuple(PARAMETERS)
FOLDS = tuple((city, seed) for city in CITIES for seed in SEEDS)
SCHEMA_VERSION = "design-controls-independent-audit-v1"
ENERGY_METRICS = ("peak_magnitude_mae", "ramp_mae", "total_24h_mae",
                  "peak_timing_mae_hours")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_record(a: np.ndarray) -> dict:
    h = hashlib.sha256()
    if a.ndim:
        for start in range(0, len(a), 8):
            h.update(np.ascontiguousarray(a[start:start + 8]).tobytes())
    else:
        h.update(a.tobytes())
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": h.hexdigest()}


def file_record(path: Path) -> dict:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"artifact changed while hashing: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dotted(data: dict, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(data, dict) or key not in data:
            raise KeyError(path)
        data = data[key]
    return data


def field(data: dict, name: str, aliases: tuple[str, ...], schema: dict) -> Any:
    paths = (schema[name],) if name in schema else aliases
    found = []
    for path in paths:
        try:
            found.append((path, dotted(data, path)))
        except KeyError:
            pass
    require(found, f"missing metadata {name}; tried {paths}; provide --schema-map for the actual schema")
    if len(found) > 1:
        require(all(value == found[0][1] for _, value in found), f"conflicting aliases for {name}: {found}")
    return found[0][1]


def canonical_path(project: Path, city: str, seed: int) -> Path:
    fold = f"target_{city.lower()}_seed{seed}"
    bases = ('experiments/forecasting/h24_framework',
             "experiments/supplementary/current_matched_ablation_20261027_homogeneous")
    candidates = [project / base / fold / "predictions.npz" for base in bases]
    return next((p for p in candidates if p.is_file()), candidates[0])


def read_fold(path: Path, canonical: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        required = {"truth", "label_starts", "site_ids", "origin_time", "target_time"}
        require(required <= set(z.files), f"fold_data missing {required - set(z.files)}")
        arrays = {key: z[key] for key in required}
        if "truth_raw" in z.files:
            require(np.array_equal(arrays["truth"], z["truth_raw"]), "truth/truth_raw alias mismatch")
    y, starts, sites = (arrays[k] for k in ("truth", "label_starts", "site_ids"))
    require(y.ndim == 3 and y.shape[:2] == (721, 24), f"truth must be (721,24,sites): {y.shape}")
    require(y.dtype == np.float32 and np.isfinite(y).all(), "truth must be finite FP32 native units")
    require(starts.dtype.kind in "iu" and np.array_equal(starts, np.arange(2928, 3649)),
            "label_starts must be all 721 integers 2928..3648")
    require(sites.ndim == 1 and len(sites) == y.shape[2] and len(np.unique(sites)) == len(sites),
            "site IDs must be unique and match truth station dimension")
    with np.load(canonical, allow_pickle=False) as old:
        for key, canonical_key in (("truth", "truth_raw"), ("label_starts", "label_starts"), ("site_ids", "site_ids")):
            reference = old[canonical_key]
            require(np.array_equal(arrays[key], reference), f"historical canonical {key}/order mismatch: {canonical}")
    expected = np.datetime64("2023-04-01T00", "h") + starts.astype("timedelta64[h]")
    times = {}
    for key in ("origin_time", "target_time"):
        a = arrays[key]
        require(a.dtype.kind in "USM", f"{key} must use datetime64 or ISO strings; got {a.dtype}")
        times[key] = a.astype("datetime64[ns]")
        require(not np.isnat(times[key]).any(), f"NaT in {key}")
    require(times["origin_time"].shape == (721,), "origin_time must have shape (721,)")
    require(times["target_time"].shape == (721, 24), "target_time must have shape (721,24)")
    targets = expected[:, None] + np.arange(24).astype("timedelta64[h]")[None, :]
    require(np.array_equal(times["target_time"], targets), "target_time does not match label starts / forecast leads")
    # Origin may denote either forecast issue time or first label time; record it.
    origin_offset = None
    for offset in (-1, 0):
        if np.array_equal(times["origin_time"], expected + np.timedelta64(offset, "h")):
            origin_offset = offset
    require(origin_offset is not None, "origin_time is neither issue hour nor first target hour")
    return y, {"arrays": {k: array_record(a) for k, a in arrays.items()},
               "origin_offset_from_label_hours": origin_offset, "canonical": file_record(canonical),
               "file": file_record(path)}


def load_energy_evaluator(project: Path):
    path = project / "experiments/energy_value/run_energy_evaluation.py"
    if not path.is_file():
        return None, {"implementation": "independent FP64 city trajectory fallback"}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "origin_scores"]
        require(len(nodes) == 1, "origin_scores unavailable")
        scope = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        return scope["origin_scores"], {"implementation": "existing evaluator pure origin_scores definition (AST extraction; no module import)", "file": file_record(path)}
    except (OSError, SyntaxError, ImportError, AttributeError, ValueError) as e:
        return None, {"implementation": "independent FP64 city trajectory fallback", "unavailable_reason": str(e),
                      "file": file_record(path)}


def score_prediction(path: Path, truth: np.ndarray, chunk: int, origin_scores=None) -> dict:
    p = np.load(path, mmap_mode="r", allow_pickle=False)
    require(isinstance(p, np.ndarray) and p.shape == truth.shape, f"prediction shape mismatch: {getattr(p, 'shape', None)}")
    require(p.dtype == np.float32, f"canonical prediction dtype must be FP32: {p.dtype}")
    sums = {mode: [0.0, 0.0] for mode in ("raw", "nonnegative_clip")}
    city_y = np.empty(truth.shape[:2], dtype=np.float64)
    city_p = {mode: np.empty_like(city_y) for mode in sums}
    for start in range(0, len(p), chunk):
        end = start + chunk
        predicted = p[start:end].astype(np.float64)
        actual = truth[start:end].astype(np.float64)
        require(np.isfinite(predicted).all(), f"nonfinite prediction at origins {start}:{end}")
        city_y[start:end] = actual.sum(axis=2, dtype=np.float64)
        for mode in sums:
            pp = predicted if mode == "raw" else np.maximum(predicted, 0.0)
            error = pp - actual
            sums[mode][0] += float(np.sum(error * error, dtype=np.float64))
            sums[mode][1] += float(np.sum(np.abs(error), dtype=np.float64))
            city_p[mode][start:end] = pp.sum(axis=2, dtype=np.float64)
    result = {}
    for mode, (sse, sae) in sums.items():
        cp = city_p[mode]
        if origin_scores is not None:
            scores = origin_scores(city_y, cp)
            energy = {key: float(np.mean(scores[key], dtype=np.float64)) for key in ENERGY_METRICS}
        else:
            energy = {"peak_magnitude_mae": float(np.abs(cp.max(1) - city_y.max(1)).mean()),
                      "ramp_mae": float(np.abs(np.diff(cp, axis=1) - np.diff(city_y, axis=1)).mean()),
                      "total_24h_mae": float(np.abs(cp.sum(1) - city_y.sum(1)).mean()),
                      "peak_timing_mae_hours": float(np.abs(cp.argmax(1) - city_y.argmax(1)).mean())}
        result[mode] = {"rmse": math.sqrt(sse / truth.size), "mae": sae / truth.size, **energy}
        require(all(math.isfinite(v) for v in result[mode].values()), "nonfinite recomputed metric")
    return {"array": array_record(p), "metrics": result}


def checkpoint_record(path: Path, variant: str) -> dict:
    import torch
    # Safe tensor-only unpickling; never fall back to importing unknown classes.
    # The runner saves NumPy's uint32 RNG array alongside tensors. Allow only
    # these NumPy reconstruction types, never arbitrary checkpoint classes.
    numpy_multiarray = importlib.import_module("numpy.core.multiarray")
    safe = [numpy_multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(safe):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    def finite_tree(value, name):
        if isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all().item()), f"nonfinite checkpoint state {name}")
        elif isinstance(value, dict):
            for key, item in value.items():
                finite_tree(item, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                finite_tree(item, f"{name}[{i}]")
        elif isinstance(value, np.ndarray) and value.dtype.kind in "fc":
            require(np.isfinite(value).all(), f"nonfinite checkpoint array {name}")
        elif isinstance(value, float):
            require(math.isfinite(value), f"nonfinite checkpoint scalar {name}")
    finite_tree(saved, "checkpoint")
    require(saved.get("variant") == variant, "checkpoint variant identity mismatch")
    state = saved
    for key in ("model_state_dict", "state_dict", "model_state", "model"):
        if isinstance(saved, dict) and isinstance(saved.get(key), dict):
            state = saved[key]
            break
    require(isinstance(state, dict) and state, "checkpoint has no tensor state dictionary")
    tensors = {}
    parameters = 0
    buffers = ("running_mean", "running_var", "num_batches_tracked")
    for name, tensor in state.items():
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor), f"non-tensor state entry {name}")
        require(bool(torch.isfinite(tensor).all().item()), f"nonfinite checkpoint tensor {name}")
        is_buffer = name.rsplit(".", 1)[-1] in buffers
        if tensor.is_floating_point() and not is_buffer:
            require(tensor.dtype == torch.float32, f"non-FP32 checkpoint parameter {name}: {tensor.dtype}")
            parameters += tensor.numel()
        tensors[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(), "buffer": is_buffer,
                         "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    require(parameters == PARAMETERS[variant], f"parameter count {parameters} != {PARAMETERS[variant]} for {variant}")
    return {"parameter_count": parameters, "tensors": tensors,
            "initialization": saved.get("initialization") if isinstance(saved, dict) else None,
            "fold_protocol": saved.get("fold_protocol") if isinstance(saved, dict) else None,
            "source_hashes": saved.get("source_hashes") if isinstance(saved, dict) else None,
            "loader": "torch.load(weights_only=True,map_location=cpu; explicit NumPy RNG allowlist)"}


def resolve_source(path_text: str, root: Path, project: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    existing = [p for p in (project / path, root / path, root / "code_snapshots" / path) if p.is_file()]
    require(existing, f"source/protocol path cannot be resolved: {path_text}")
    return existing[0]


def hash_entries(value: Any) -> list[tuple[str, str]]:
    require(isinstance(value, dict) and value, "source hashes must be a nonempty path-to-SHA256 mapping")
    entries = []
    for path, digest in value.items():
        if isinstance(digest, dict):
            path, digest = digest.get("path", path), digest.get("sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest), f"invalid hash for {path}")
        entries.append((path, digest.lower()))
    return entries


def metadata_record(data: dict, city: str, seed: int, variant: str, root: Path, project: Path, schema: dict, run_dir: Path) -> dict:
    get = lambda key, *aliases: field(data, key, aliases, schema)
    status = get("status", "status")
    require(status in ("completed", "complete", "success"), f"not a full completed run: {status}")
    require(str(get("city", "target", "city", "target_city")).upper() == city, "result city identity mismatch")
    require(int(get("seed", "seed")) == seed, "result seed identity mismatch")
    require(get("variant", "variant", "model_variant", "model") == variant, "result variant identity mismatch")
    for key, expected, aliases in (
        ("minibatches", 2048, ("steps", "configured_source_minibatches", "training.steps", "training.source_minibatches", "training.configured_source_minibatches", "budget.source_minibatches", "protocol.source_minibatches")),
        ("windows", 8192, ("window_exposures", "training.effective_source_windows", "training.source_windows", "budget.effective_source_windows", "protocol.effective_source_windows")),
        ("updates", 410, ("optimizer_steps", "training.pcgrad_rounds", "training.optimizer_steps", "training.pcgrad_updates", "budget.pcgrad_rounds", "protocol.pcgrad_rounds")),
        ("batch_size", 4, ("training.batch_size_windows", "training.batch_size", "budget.batch_size_windows", "protocol.batch_size"))):
        observed = field(data, key, aliases, schema)
        require(observed == expected and not isinstance(observed, bool), f"{key}: {observed} != {expected}")
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    precision_data = {**data, "run_config": config}
    precision = str(field(precision_data, "precision", ("training.precision", "precision", "protocol.precision", "training.dtype", "run_config.precision"), schema)).lower()
    require(precision in ("fp32", "float32", "torch.float32", "float32; no amp"), f"noncanonical precision: {precision}")
    for container in (data.get("training", {}), data.get("protocol", {})):
        for key in ("amp", "use_amp", "mixed_precision"):
            if key in container:
                require(container[key] in (False, None, "disabled", "off"), f"mixed precision enabled: {key}")
    protocol = get("protocol", "fold_protocol", "protocol")
    require(isinstance(protocol, dict) and protocol, "protocol metadata absent")
    protocol_hash = get("protocol_sha256", "protocol_sha256", "protocol.protocol_sha256", "protocol.frozen_protocol_sha256")
    require(isinstance(protocol_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", protocol_hash), "invalid protocol SHA256")
    sources = get("source_hashes", "source_sha256", "source_hashes", "source_hashes_sha256", "provenance.source_sha256")
    records, transient = [], []
    nontraining_scripts = {"evaluate_controls.py", "audit_completed_runs.py", "update_matrices.py"}
    for source_path, expected_hash in hash_entries(sources):
        if Path(source_path).name in nontraining_scripts:
            # These scripts can be finalized after the model/runner freeze.
            # Their transient presence in sys.modules is not training lineage.
            transient.append({"declared_path": source_path, "declared_sha256": expected_hash,
                              "audit_role": "informational non-training script"})
            continue
        path = resolve_source(source_path, root, project)
        record = file_record(path)
        if record["sha256"] != expected_hash:
            snapshots = root / "code_snapshots"
            matches = [p for p in snapshots.rglob(path.name) if p.is_file() and sha256_file(p) == expected_hash]
            require(matches, f"source hash mismatch with no matching frozen snapshot: {path}")
            record = file_record(sorted(matches)[0])
        record["declared_source"] = str(path.resolve())
        require(record["sha256"] == expected_hash, f"source hash mismatch: {path}")
        records.append(record)
    protocol_path = project / "scprc/training/scprc_matched_protocol.json"
    require(sha256_file(protocol_path) == protocol_hash.lower(), "frozen protocol file hash mismatch")
    require(any(Path(r["declared_source"]) == protocol_path.resolve() for r in records), "frozen protocol missing from source manifest")
    # Exact runner schema records its helper through the source manifest and
    # training.gradient_aggregation, rather than a canonical_helper key.
    helper = data.get("canonical_helper", "scprc/training/train_scprc.py")
    if "canonical_helper" in schema:
        helper = dotted(data, schema["canonical_helper"])
    require(isinstance(helper, str) and helper, "canonical helper identity missing")
    helper_path = resolve_source(helper, root, project).resolve()
    require(any(Path(r["declared_source"]) == helper_path for r in records), "canonical helper must be covered by verified source hashes")
    primary = {root / "run_control.py", root / "model_variants.py", helper_path, protocol_path.resolve(),
               project / "experiments/forecasting/h24_framework/run_canonical_anchor_fold.py"}
    by_declared = {Path(r["declared_source"]): r["sha256"] for r in records}
    require(primary <= set(by_declared), "runner/model/helper/protocol primary source hashes incomplete")
    primary_signature = {str(path): digest for path, digest in by_declared.items()
                         if path in primary or "scprc/models" in str(path)}
    require("pcgrad" in str(data.get("training", {}).get("gradient_aggregation", "")).lower(), "PCGrad aggregation metadata missing")
    require(data.get("protocol_id") == "scpcr-matched-baseline-20261023", "canonical protocol ID mismatch")
    if "parameters" in data.get("training", {}):
        require(data["training"]["parameters"] == PARAMETERS[variant], "training parameter-count metadata mismatch")
    for key, expected in {"context": 168, "horizon": 24, "source_train_rows": [0, 2928],
                          "evaluation_label_starts": [2928, 3649], "evaluation_origin_count": 721,
                          "target_support_projection": False, "target_release_gate": False,
                          "retrospective": True, "threshold_quantile": 0.9}.items():
        require(protocol.get(key) == expected, f"fold protocol {key} mismatch")
    require(protocol.get("target_city") == city and protocol.get("seed") == seed, "fold protocol identity mismatch")
    require(set(protocol.get("source_cities", [])) == set(CITIES) - {city}, "LOCO source cities mismatch")
    scaler = protocol.get("source_scaler", {})
    require(all(isinstance(scaler.get(k), (int, float)) and math.isfinite(scaler[k]) for k in ("mean", "std")) and scaler["std"] > 0, "invalid source scaler")
    if config:
        require(config.get("variant") == variant and config.get("steps") == 2048 and config.get("fold") == protocol, "run_config/result protocol mismatch")
        require(config.get("source_hashes") == sources, "run_config/result source hashes mismatch")
    return {"protocol": protocol, "protocol_sha256": protocol_hash.lower(), "source_files": records,
            "canonical_helper": str(helper_path), "precision": precision,
            "primary_source_hashes": primary_signature, "transient_source_entries": transient,
            "run_config": file_record(config_path) if config else None}


def provenance_record(run_dir: Path, data: dict, variant: str, city: str) -> dict:
    initial = data.get("initialization", {})
    for key in ("all_state_sha256", "backbone_state_sha256"):
        require(isinstance(initial.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", initial[key]), f"missing initialization {key}")
    require(initial.get("parameter_count") == PARAMETERS[variant] == data.get("parameter_count"), "initialization/result parameter count mismatch")
    initial_file = run_dir / "initialization.json"
    require(json.loads(initial_file.read_text()) == initial, "initialization.json differs from result")
    code_file = run_dir / "code_hashes.json"
    require(json.loads(code_file.read_text()) == data["source_hashes"], "code_hashes.json differs from result")
    trace_file = run_dir / "sampling_trace.json"
    trace = json.loads(trace_file.read_text())
    batches, orders = trace.get("source_batches", []), trace.get("pcgrad_orders", [])
    require(len(batches) == 2048 and len(orders) == 410, "sampler trace does not prove 2048 batches/410 updates")
    source_ids = set(CITIES) - {city}
    for item in batches:
        starts = item.get("label_starts", [])
        require(item.get("city") in source_ids and len(starts) == 4 and
                all(isinstance(s, int) and not isinstance(s, bool) and 168 <= s <= 2904 for s in starts), "invalid source-only sampled window trace")
    for update, order in enumerate(orders):
        task_count = min(5, 2048 - update * 5)
        require(isinstance(order, list) and sorted(order) == list(range(task_count)), "invalid PCGrad shuffle order")
    # Source tasks follow the canonical 409 full rounds plus a three-task round.
    source_order = data["fold_protocol"]["source_cities"]
    expected_sequence = source_order * 409 + source_order[:3]
    require([b["city"] for b in batches] == expected_sequence, "source-city exposure schedule differs from canonical PCGrad")
    # Compare semantic trace hashes, independent of JSON whitespace/key order.
    canonical = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return {"initialization": initial, "sampling_semantic_sha256": hashlib.sha256(canonical).hexdigest(),
            "sampling_trace": file_record(trace_file), "initialization_file": file_record(initial_file),
            "code_hashes_file": file_record(code_file)}


def audit(root: Path, project: Path, schema: dict | None = None, chunk: int = 8) -> dict:
    root, project = root.resolve(), project.resolve()
    schema = schema or {}
    origin_scores, evaluator = load_energy_evaluator(project)
    expected_paths = {root / "runs" / f"target_{city.lower()}_seed{seed}" / v / "result.json"
                      for city, seed in FOLDS for v in VARIANTS}
    actual_paths = set((root / "runs").glob("target_*_seed*/*/result.json"))
    errors, folds, runs = [], [], []
    if actual_paths != expected_paths:
        errors.append({"scope": "run_counts", "missing": [str(p) for p in sorted(expected_paths - actual_paths)],
                       "unexpected": [str(p) for p in sorted(actual_paths - expected_paths)]})
    protocol_hashes, source_signatures = set(), set()
    for city, seed in FOLDS:
        fold_dir = root / "runs" / f"target_{city.lower()}_seed{seed}"
        try:
            truth, alignment = read_fold(fold_dir / "fold_data.npz", canonical_path(project, city, seed))
            folds.append({"city": city, "seed": seed, **alignment})
        except Exception as e:
            errors.append({"scope": str(fold_dir), "error": f"{type(e).__name__}: {e}"})
            continue
        for variant in VARIANTS:
            run_dir = fold_dir / variant
            row = {"city": city, "seed": seed, "variant": variant, "passed": False}
            try:
                result_path = run_dir / "result.json"
                before = file_record(result_path)
                data = json.loads(result_path.read_text(encoding="utf-8"))
                meta = metadata_record(data, city, seed, variant, root, project, schema, run_dir)
                provenance = provenance_record(run_dir, data, variant, city)
                require(meta["protocol"].get("fold_data_sha256") == alignment["file"]["sha256"], "fold_data declared hash mismatch")
                protocol_hashes.add(meta["protocol_sha256"])
                source_signatures.add(tuple(sorted(meta["primary_source_hashes"].items())))
                pred_file = file_record(run_dir / "prediction_raw.npy")
                ckpt_file = file_record(run_dir / "checkpoint.pt")
                for key, record, aliases in (
                    ("prediction_sha256", pred_file, ("predictions_sha256", "prediction_sha256", "prediction_file_sha256", "artifacts.prediction_sha256", "artifacts.prediction_raw.sha256")),
                    ("checkpoint_sha256", ckpt_file, ("checkpoint_sha256", "checkpoint_file_sha256", "artifacts.checkpoint_sha256", "artifacts.checkpoint.sha256"))):
                    require(field(data, key, aliases, schema) == record["sha256"], f"declared {key} mismatch")
                for key, actual in (("checkpoint", ckpt_file), ("predictions", pred_file)):
                    declared = Path(data[key])
                    if not declared.is_absolute():
                        declared = resolve_source(data[key], root, project)
                    require(declared.resolve() == Path(actual["path"]), f"result {key} path mismatch")
                prediction = score_prediction(run_dir / "prediction_raw.npy", truth, chunk, origin_scores)
                state = checkpoint_record(run_dir / "checkpoint.pt", variant)
                require(state["initialization"] == provenance["initialization"], "checkpoint/result initialization mismatch")
                checkpoint_protocol = dict(state["fold_protocol"])
                if isinstance(checkpoint_protocol.get("source_cities"), tuple):
                    # torch.save preserves tuples; JSON encodes ordered tuples
                    # as arrays. Normalize only this known ordered ID sequence.
                    checkpoint_protocol["source_cities"] = list(checkpoint_protocol["source_cities"])
                    state["protocol_serialization_normalization"] = "source_cities ordered tuple -> JSON list; no values or order changed"
                require(checkpoint_protocol == meta["protocol"], "checkpoint/result fold protocol mismatch")
                require(state["source_hashes"] == data["source_hashes"], "checkpoint/result source hash mismatch")
                for metric in ("rmse", "mae"):
                    stored = field(data, metric, (f"metrics.{metric}", f"metrics.raw.{metric}", f"metrics.native_{metric}"), schema)
                    require(isinstance(stored, (int, float)) and math.isfinite(stored), f"invalid stored {metric}")
                    require(math.isclose(stored, prediction["metrics"]["raw"][metric], rel_tol=1e-6, abs_tol=1e-6),
                            f"stored/recomputed native {metric} mismatch: {stored} vs {prediction['metrics']['raw'][metric]}")
                for record in (before, pred_file, ckpt_file):
                    require(file_record(Path(record["path"]))["sha256"] == record["sha256"], f"artifact changed during audit: {record['path']}")
                row.update(passed=True, result=before, prediction_file=pred_file, checkpoint_file=ckpt_file,
                           prediction=prediction, checkpoint=state, metadata=meta, provenance=provenance)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                errors.append({"scope": str(run_dir), "error": row["error"]})
            runs.append(row)
        selected = [r for r in runs if r["city"] == city and r["seed"] == seed and r["passed"]]
        if len(selected) == 4:
            try:
                require(len({r["provenance"]["initialization"]["backbone_state_sha256"] for r in selected}) == 1, "four variants have different backbone initialization hashes")
                require(len({r["provenance"]["sampling_semantic_sha256"] for r in selected}) == 1, "four variants have different sampler/PCGrad order traces")
                by_variant = {r["variant"]: r for r in selected}
                require(by_variant["full_same_run"]["provenance"]["initialization"]["all_state_sha256"] ==
                        by_variant["phase_shared_strict"]["provenance"]["initialization"]["all_state_sha256"], "Full/phase all-state initialization hash mismatch")
                require(all(r["metadata"]["protocol"] == selected[0]["metadata"]["protocol"] for r in selected), "four variants have different fold protocols")
                for key in ("profile_sha256", "query_sha256", "key_sha256", "profile_qk_sha256"):
                    common = [by_variant[v]["provenance"]["initialization"].get(key) for v in ("full_same_run", "phase_shared_strict", "concat_project")]
                    if any(v is not None for v in common):
                        require(None not in common and len(set(common)) == 1, f"shared initial {key} mismatch")
                folds[-1]["four_variant_initialization_and_sampling_pass"] = True
            except Exception as e:
                errors.append({"scope": str(fold_dir), "error": str(e)})
                folds[-1]["four_variant_initialization_and_sampling_pass"] = False
                for r in selected:
                    r["passed"] = False
                    r["error"] = str(e)
    if len(protocol_hashes) != 1:
        errors.append({"scope": "protocol", "error": f"expected one shared protocol hash; found {len(protocol_hashes)}"})
    if len(source_signatures) != 1:
        errors.append({"scope": "source_hashes", "error": f"expected one primary model/runner/helper source manifest; found {len(source_signatures)}"})
    observer = None
    observer_path = root / "observer_parity_test.json"
    if observer_path.is_file():
        try:
            evidence = json.loads(observer_path.read_text())
            require(evidence.get("passed") is True and evidence.get("all_state_tensors_bitwise_equal") is True, "observer parity evidence reports failure")
            observer = {"file": file_record(observer_path), "evidence": evidence}
        except (OSError, ValueError) as error:
            errors.append({"scope": str(observer_path), "error": str(error)})
    summaries = {}
    for variant in VARIANTS:
        selected = [r for r in runs if r["variant"] == variant and r["passed"]]
        if len(selected) == 18:
            summaries[variant] = {"fold_count": 18, "parameter_count": PARAMETERS[variant],
                "metrics": {mode: {k: float(np.mean([r["prediction"]["metrics"][mode][k] for r in selected]))
                    for k in ("rmse", "mae", *ENERGY_METRICS)} for mode in ("raw", "nonnegative_clip")}}
    passed = not errors and len(runs) == 72 and all(r["passed"] for r in runs) and len(folds) == 18 and len(summaries) == 4
    return {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "project_root": str(project), "full_audit_pass": passed,
            "status": "passed" if passed else "failed", "expected_runs": 72,
            "discovered_runs": len(actual_paths), "passed_runs": sum(r["passed"] for r in runs),
            "expected_folds": 18, "audited_folds": len(folds), "schema_map": schema,
            "auditor": file_record(Path(__file__)), "energy_evaluator": evaluator,
            "observer_parity": observer,
            "model_summaries": summaries, "folds": folds, "runs": runs, "errors": errors,
            "aggregation": "arithmetic mean of all 18 fold metrics; native FP64; raw and clipping separate"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--project-root", type=Path, default=PROJECT)
    p.add_argument("--output", type=Path, help="default ROOT/audit/completed_runs_audit.json")
    p.add_argument("--schema-map", type=Path, help="JSON logical-name -> dotted result path (e.g. updates: training.optimizer_steps); no checks bypassed")
    p.add_argument("--chunk-origins", type=int, default=8)
    args = p.parse_args(argv)
    if args.chunk_origins < 1:
        p.error("--chunk-origins must be positive")
    schema = json.loads(args.schema_map.read_text()) if args.schema_map else {}
    report = audit(args.root, args.project_root, schema, args.chunk_origins)
    output = args.output or args.root / "audit/completed_runs_audit.json"
    atomic_json(output, report)
    print(json.dumps({"audit": str(output), "full_audit_pass": report["full_audit_pass"],
                      "passed_runs": report["passed_runs"], "errors": len(report["errors"])}))
    return 0 if report["full_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

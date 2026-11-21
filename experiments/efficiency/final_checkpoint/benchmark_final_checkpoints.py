#!/usr/bin/env python3
"""Synchronized CUDA efficiency benchmark for all frozen CHARGED Exact/SCPRC folds.

Profiles the final 18 city x seed checkpoints at batch sizes 1, 4 and 16.
Only model-forward time is measured; preprocessing and host/device copies are
outside the timed region. Inputs are target-city August histories standardized
with the source-only scaler for the corresponding held-out city.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, platform, statistics, subprocess, sys, time, types
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
FOLD_ROOT = Path(os.environ.get(
    "REBUILD_FOLD_ROOT",
    str(ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"),
))
OUT = Path(os.environ.get("REBUILD_EFFICIENCY_OUT", str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True, exist_ok=True)
for entry in (ROOT,):
    sys.path.insert(0, str(entry))
import scprc.training.training_protocol as protocol


def project_artifact_path(value):
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scprc.project_paths import project_artifact_path as resolve
    return resolve(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def imports():
    legacy_name = "calibration_protocol_constants"
    if legacy_name not in sys.modules:
        legacy = types.ModuleType(legacy_name)
        legacy.CONTEXT = protocol.CONTEXT
        legacy.HORIZON = protocol.HORIZON
        legacy.TRAIN_END = protocol.TRAIN_END
        legacy.protocol = protocol
        sys.modules[legacy_name] = legacy
    import scprc.training.scprc_training_support as adapter
    return adapter


def env(device: torch.device) -> dict[str, Any]:
    out = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
           "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
           "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
           "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)}
    if device.type == "cuda":
        out.update({"cuda_available": True, "device_name": torch.cuda.get_device_name(device),
                    "device_capability": list(torch.cuda.get_device_capability(device)),
                    "device_properties": str(torch.cuda.get_device_properties(device))})
        try:
            out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                  "--format=csv,noheader"], capture_output=True, text=True,
                                                 check=False).stdout.strip()
        except Exception as e:
            out["nvidia_smi"] = str(e)
    return out


def load_batch(cities: dict[str, Any], target: str, seed: int, batch_size: int, device: torch.device):
    from scprc.data_adapters.charged.city_adapter import CITY_CODES
    sources = tuple(c for c in CITY_CODES if c != target)
    mean, std = protocol.source_scaler(cities, sources)
    origins = np.arange(protocol.TRAIN_END, protocol.VAL_END - protocol.HORIZON + 1, dtype=np.int64)
    picks = np.random.default_rng(seed).choice(origins, size=batch_size, replace=False)
    history = np.stack([cities[target]["values"][int(o) - protocol.CONTEXT:int(o)] for o in picks]).astype(np.float32)
    x = torch.from_numpy((history - mean) / std).to(device)
    return x.contiguous(), {"target_city": target, "seed": seed, "origins": [int(x) for x in picks],
                            "input_shape": list(x.shape), "source_scaler_fit_cities": list(sources),
                            "source_scaler_fit_interval": "[0, 2928), 2023-04-01 through 2023-07-31",
                            "context_hours": protocol.CONTEXT, "horizon_hours": protocol.HORIZON,
                            "preprocessing_in_timed_region": False}


def p95(vals):
    return sorted(vals)[min(len(vals) - 1, max(0, math.ceil(.95 * len(vals)) - 1))]


def benchmark(model, x, device, warmup=12, repeats=80):
    model.eval()
    with torch.no_grad():
        y = model(x)
    expected = (x.shape[0], protocol.HORIZON, x.shape[2])
    if tuple(y.shape) != expected or not torch.isfinite(y).all():
        raise RuntimeError(f"invalid output shape/values: {tuple(y.shape)} expected {expected}")
    with torch.no_grad():
        for _ in range(warmup):
            model(x); torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    values = []
    with torch.no_grad():
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter(); model(x); torch.cuda.synchronize(device)
            values.append(time.perf_counter() - t0)
    med = statistics.median(values)
    return {"warmup": warmup, "repeats": repeats, "p50_ms_batch": med * 1000,
            "p95_ms_batch": p95(values) * 1000, "mean_ms_batch": statistics.mean(values) * 1000,
            "throughput_windows_per_second": x.shape[0] / med,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
            "precision": "FP32", "cuda_synchronized": True, "output_shape": list(y.shape)}


def resolve_fold_artifact(fold: Path, result: dict[str, Any], arm: str) -> Path:
    """Resolve both the historical flat and the rebuilt nested checkpoint layouts."""
    flat_name = "exact_pcgrad.pt" if arm == "exact" else "scpcr_candidate.pt"
    flat = fold / flat_name
    if flat.is_file():
        return flat
    rebuilt_key = "exact_pcgrad" if arm == "exact" else "scpcr_candidate"
    checkpoint = result.get("checkpoints", {}).get(rebuilt_key)
    if checkpoint:
        nested = project_artifact_path(checkpoint)
        if nested.is_file():
            return nested
    nested = fold / rebuilt_key / "checkpoint.pt"
    if nested.is_file():
        return nested
    return flat


def result_training_record(result: dict[str, Any], arm: str) -> dict[str, Any]:
    """Read threshold metadata from either canonical result schema."""
    key = "exact_pcgrad" if arm == "exact" else "scpcr_candidate"
    record = result.get("training", {}).get(key, {})
    if record:
        return record
    return result.get("training", {}).get("scpcr_candidate", {})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    adapter = imports()
    cities, provenance = protocol.load_cities()
    rows, input_records, checkpoint_records = [], [], []
    cities_order = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
    seeds = (13, 17, 29)
    for target in cities_order:
        channels = len(cities[target]["site_ids"])
        for seed in seeds:
            fold = FOLD_ROOT / f"target_{target.lower()}_seed{seed}"
            result_path = fold / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"missing fold result: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            training_candidate = result_training_record(result, "full")
            threshold = float(
                training_candidate.get(
                    "source_relation_threshold",
                    result.get("source_relation_threshold", 0.9),
                )
            )
            support = float(training_candidate.get("support_scale", 1.0))
            specs = [("PhaseFormer Exact", protocol._build_model("phaseformer_exact", channels), fold / "exact_pcgrad.pt"),
                     ("SCPRC Full", adapter._build_model(channels, threshold, support), fold / "scpcr_candidate.pt")]
            specs[0] = (specs[0][0], specs[0][1], resolve_fold_artifact(fold, result, "exact"))
            specs[1] = (specs[1][0], specs[1][1], resolve_fold_artifact(fold, result, "full"))
            inputs = {}
            for bs in (1, 4, 16):
                x, prov = load_batch(cities, target, seed, bs, device)
                inputs[bs] = x
                if bs == 16: input_records.append(prov)
            for name, model, ckpt in specs:
                if not ckpt.is_file():
                    checkpoint_records.append({"target_city": target, "seed": seed, "model": name,
                                               "checkpoint": str(ckpt), "status": "missing_checkpoint",
                                               "reason": "checkpoint is absent from the selected fold root"})
                    del model
                    continue
                payload = torch.load(ckpt, map_location="cpu", weights_only=True)
                state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                model.load_state_dict(state, strict=True)
                model = model.float().to(device).eval()
                params = int(sum(p.numel() for p in model.parameters()))
                checkpoint_records.append({"target_city": target, "seed": seed, "model": name,
                                           "checkpoint": str(ckpt.relative_to(ROOT)), "sha256": sha256(ckpt),
                                           "bytes": ckpt.stat().st_size, "parameters": params,
                                           "source_relation_threshold": threshold, "source_support_passed": bool(support)})
                for bs in (1, 4, 16):
                    stat = benchmark(model, inputs[bs], device)
                    rows.append({"target_city": target, "seed": seed, "model": name, "batch_size": bs,
                                 "site_count": channels, "parameters": params, "checkpoint": str(ckpt.relative_to(ROOT)),
                                 "checkpoint_sha256": sha256(ckpt), **stat})
                del model
                torch.cuda.empty_cache()
            print(f"completed {target}/seed{seed} ({channels} sites)", flush=True)
    if not rows:
        raise RuntimeError("no checkpoint measurements were produced")
    with (OUT / "final_checkpoint_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    record = {"status": "completed", "generated": "2026-10-30", "scope": "selected canonical CHARGED folds; synchronized CUDA model-forward only",
              "canonical_prediction_lineage": str(FOLD_ROOT.relative_to(ROOT)),
              "device": env(device), "dataset_provenance": provenance, "folds": [f"{c}/seed{s}" for c in cities_order for s in seeds],
              "batch_sizes": [1, 4, 16], "models": ["PhaseFormer Exact", "SCPRC Full"], "warmup": 12, "repeats": 80,
              "input_records_batch16": input_records, "checkpoint_records": checkpoint_records, "rows": rows,
              "script_sha256": sha256(Path(__file__).resolve())}
    (OUT / "final_checkpoint_efficiency.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "environment.json").write_text(json.dumps(env(device), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "completed", "rows": len(rows), "output": str(OUT)}, ensure_ascii=False))

if __name__ == "__main__": main()

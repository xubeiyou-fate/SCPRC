#!/usr/bin/env python3
"""Run the existing synchronized CUDA benchmark for Original, Exact, and Full.

The model construction, input preparation, timing, and CUDA synchronization are
reused from experiments/efficiency/final_checkpoint/benchmark_final_checkpoints.py.
Only the input roots and output directory are changed; old measurements are
never modified.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/efficiency/final_checkpoint/benchmark_final_checkpoints.py"
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
FULL_ROOT = ROOT / 'experiments/forecasting/h24_framework'
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)


def project_artifact_path(value):
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scprc.project_paths import project_artifact_path as resolve
    return resolve(value)


def load_source():
    spec = importlib.util.spec_from_file_location("canonical_efficiency_benchmark", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_source()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the efficiency benchmark")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    protocol = source.protocol
    adapter = source.imports()
    cities, provenance = protocol.load_cities()
    rows = []
    input_records = []
    checkpoint_records = []
    for target in CITIES:
        channels = len(cities[target]["site_ids"])
        for seed in SEEDS:
            original_fold = ORIGINAL_ROOT / f"target_{target.lower()}_seed{seed}"
            full_fold = FULL_ROOT / f"target_{target.lower()}_seed{seed}"
            original_result = json.loads((original_fold / "result.json").read_text(encoding="utf-8"))
            full_result = json.loads((full_fold / "result.json").read_text(encoding="utf-8"))
            full_checkpoint = project_artifact_path(full_result["checkpoints"]["scpcr_candidate"])
            original_checkpoint = project_artifact_path(original_result["checkpoint"])
            specs = [
                ("Original PhaseFormer", protocol._build_model("phaseformer_exact", channels), original_checkpoint, original_result),
                ("PhaseFormer Exact-PCGrad", protocol._build_model("phaseformer_exact", channels), full_fold / "exact_pcgrad" / "checkpoint.pt", full_result),
                ("Full SCPRC", adapter._build_model(channels, float(full_result["training"]["scpcr_candidate"].get("source_relation_threshold", full_result.get("source_relation_threshold", 0.9))), float(full_result["training"]["scpcr_candidate"].get("support_scale", 1.0))), full_checkpoint, full_result),
            ]
            inputs = {}
            for batch_size in (1, 4, 16):
                x, provenance_record = source.load_batch(cities, target, seed, batch_size, device)
                inputs[batch_size] = x
                if batch_size == 16:
                    input_records.append(provenance_record)
            for name, model, checkpoint, result in specs:
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing {name} checkpoint: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                model.load_state_dict(state, strict=True)
                model = model.float().to(device).eval()
                parameters = int(sum(parameter.numel() for parameter in model.parameters()))
                checkpoint_records.append({
                    "target_city": target,
                    "seed": seed,
                    "model": name,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": source.sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "parameters": parameters,
                    "protocol_id": result.get("protocol_id"),
                })
                for batch_size in (1, 4, 16):
                    measurement = source.benchmark(model, inputs[batch_size], device)
                    rows.append({
                        "target_city": target,
                        "seed": seed,
                        "model": name,
                        "batch_size": batch_size,
                        "site_count": channels,
                        "parameters": parameters,
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "checkpoint_sha256": source.sha256(checkpoint),
                        **measurement,
                    })
                del model
                torch.cuda.empty_cache()
            print(f"completed {target}/seed{seed} ({channels} sites)", flush=True)
    if len(rows) != 18 * 3 * 3:
        raise RuntimeError(f"expected 162 rows, found {len(rows)}")
    output_csv = OUT / "efficiency_three_models.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        summary[model] = {}
        for batch_size in (1, 4, 16):
            selected = [row for row in rows if row["model"] == model and row["batch_size"] == batch_size]
            summary[model][str(batch_size)] = {
                "folds": len(selected),
                "parameters": selected[0]["parameters"],
                "p50_ms_batch_mean_over_folds": sum(float(row["p50_ms_batch"]) for row in selected) / len(selected),
                "p95_ms_batch_mean_over_folds": sum(float(row["p95_ms_batch"]) for row in selected) / len(selected),
                "throughput_windows_per_second_mean_over_folds": sum(float(row["throughput_windows_per_second"]) for row in selected) / len(selected),
                "peak_cuda_allocated_mb_max": max(float(row["peak_cuda_allocated_mb"]) for row in selected),
                "peak_cuda_reserved_mb_max": max(float(row["peak_cuda_reserved_mb"]) for row in selected),
            }
    manifest = {
        "status": "completed",
        "experiment": "E1 Original PhaseFormer efficiency supplement",
        "input_lineage_check": "lineage_check.json",
        "canonical_original_root": str(ORIGINAL_ROOT.relative_to(ROOT)),
        "canonical_full_root": str(FULL_ROOT.relative_to(ROOT)),
        "source_benchmark": str(SOURCE.relative_to(ROOT)),
        "models": ["Original PhaseFormer", "PhaseFormer Exact-PCGrad", "Full SCPRC"],
        "folds": 18,
        "rows": len(rows),
        "batch_sizes": [1, 4, 16],
        "conditions": {"precision": "FP32", "cuda_synchronized": True, "warmup": 12, "repeats": 80, "timed_region": "model forward only; preprocessing and host/device copies excluded"},
        "device": source.env(device),
        "dataset_provenance": provenance,
        "input_records_batch16": input_records,
        "checkpoint_records": checkpoint_records,
        "summary": summary,
        "old_results_overwritten": False,
    }
    (OUT / "efficiency_three_models.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# E1 efficiency: Original / Exact-PCGrad / Full",
        "",
        "Status: **completed**. Measurements use the existing synchronized CUDA benchmark implementation, with all three models evaluated under the same device, FP32, batch sizes, warm-up, and repeats.",
        "",
        "| Model | Batch | Folds | Parameters | Mean p50 ms | Mean p95 ms | Mean throughput windows/s | Max reserved MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("Original PhaseFormer", "PhaseFormer Exact-PCGrad", "Full SCPRC"):
        for batch_size in (1, 4, 16):
            row = summary[model][str(batch_size)]
            lines.append(f"| {model} | {batch_size} | {row['folds']} | {row['parameters']} | {row['p50_ms_batch_mean_over_folds']:.4f} | {row['p95_ms_batch_mean_over_folds']:.4f} | {row['throughput_windows_per_second_mean_over_folds']:.2f} | {row['peak_cuda_reserved_mb_max']:.2f} |")
    lines += [
        "",
        "The benchmark measures forward inference only. It does not claim training-time cost, end-to-end data-loader latency, or deployment cost outside this hardware/protocol.",
        "",
        "Source: `experiments/efficiency/final_checkpoint/benchmark_final_checkpoints.py`; old benchmark outputs are unchanged.",
    ]
    (OUT / "efficiency_three_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "rows": len(rows), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

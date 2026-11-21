#!/usr/bin/env python3
"""Run one source-only current-SCPRC external city fold in an isolated directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
OUTPUT = ROOT / "experiments/supplementary/external_current_20261024"
for path in (ROOT,):
    sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compatibility_modules():
    legacy_name = "calibration_protocol_constants"
    if legacy_name not in sys.modules:
        legacy = types.ModuleType(legacy_name)
        legacy.CONTEXT = protocol.CONTEXT
        legacy.HORIZON = protocol.HORIZON
        legacy.TRAIN_END = protocol.TRAIN_END
        legacy.protocol = protocol
        sys.modules[legacy_name] = legacy
    import scprc.training.scprc_training_support as v21
    import scprc.training.train_scprc as v22
    return v21, v22


def stamp_result(path: Path, seed: int, dataset: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    prediction_files = [path.parent / name for name in
                        ("predictions.npz", "q4_predictions.npz")
                        if (path.parent / name).is_file()]
    checkpoint_hashes = {
        str(item.relative_to(path.parent)): sha256(item)
        for item in sorted(path.parent.glob("*.pt"))
    }
    result["paper2_current_run_provenance"] = {
        "status": "retrospective_retraining",
        "blind_confirmation": False,
        "dataset": dataset,
        "seed": seed,
        "source_training": "CHARGED v3 six cities, April-July only",
        "target_labels_used_for_parameter_updates": False,
        "supplement_runner_sha256": sha256(Path(__file__).resolve()),
        "phaseformer_protocol_sha256": sha256(protocol.HERE),
        "source_certified_adapter_sha256": sha256(
            ROOT / "scprc/models/scprc.py"
        ),
        "pcgrad_runner_sha256": sha256(
            EXPERIMENTS / "train_scprc.py"
        ),
        "prediction_archives_sha256": {
            item.name: sha256(item) for item in prediction_files
        },
        "checkpoint_sha256": checkpoint_hashes,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                allow_nan=False) + "\n", encoding="utf-8")
    return result


def run_jiaxing(seed: int, device: str, torch_threads: int) -> dict:
    import scprc.training.run_jiaxing_external_scprc as runner

    output_root = OUTPUT / "jiaxing"
    result = runner.run(seed, output_root=output_root, device=device,
                        torch_threads=torch_threads, steps=2048)
    return stamp_result(output_root / f"seed_{seed}" / "result.json", seed,
                        "Jiaxing")


def run_shanghai(seed: int, device: str, torch_threads: int) -> dict:
    import scprc.training.run_mp_evdata_external_scprc as runner

    previous = EXPERIMENTS / "mp_evdata_scpcr_external_20261021"
    output_root = OUTPUT / "shanghai"
    seed_dir = output_root / f"seed{seed}"
    if (seed_dir / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite {seed_dir / 'result.json'}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("protocol_freeze.json", "mp_evdata_history_prefix.npz"):
        destination = output_root / name
        if not destination.exists():
            shutil.copy2(previous / name, destination)
    runner.OUT = output_root
    result = runner.run_seed(
        seed,
        steps=2048,
        torch_threads=torch_threads,
        device=device,
        archive_path=protocol.ARCHIVE,
        nested_root=EXPERIMENTS / "charged_scpcr_v22_nested_source_calibration_20261021",
    )
    return stamp_result(output_root / f"seed{seed}" / "result.json", seed,
                        "Shanghai MP-EVData")


def run_urbanev(seed: int, device: str, torch_threads: int) -> dict:
    import scprc.training.run_urbanev_external_scprc as runner

    output_root = OUTPUT / "urbanev" / f"seed_{seed}"
    if (output_root / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite {output_root / 'result.json'}")
    output_root.mkdir(parents=True, exist_ok=False)
    runner.SEED = seed
    runner.TORCH_THREADS = torch_threads
    sys.argv = [
        str(runner.HERE),
        "--output", str(output_root),
        "--device", device,
    ]
    status = runner.main()
    if status:
        raise RuntimeError(f"UrbanEV runner returned exit code {status}")
    return stamp_result(output_root / "result.json", seed, "UrbanEV Shenzhen")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=("jiaxing", "shanghai", "urbanev"))
    parser.add_argument("--seed", required=True, type=int, choices=(13, 17, 29))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    load_compatibility_modules()
    result = {
        "jiaxing": run_jiaxing,
        "shanghai": run_shanghai,
        "urbanev": run_urbanev,
    }[args.dataset](args.seed, args.device, args.torch_threads)
    print(json.dumps({"status": result.get("status"), "dataset": args.dataset,
                      "seed": args.seed, "metrics": result.get("metrics"),
                      "result": str(OUTPUT / args.dataset)},
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train one provenance-pinned SCPRC source-only CHARGED LOCO fold."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
OUTPUT = ROOT / "experiments/supplementary/current_scpcr_charged_20261024"
for path in (ROOT,):
    sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runners():
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


def run_fold(target: str, seed: int, device: str, torch_threads: int) -> dict:
    v21, v22 = load_runners()
    target = target.upper()
    fold_dir = OUTPUT / f"target_{target.lower()}_seed{seed}"
    result_path = fold_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}
    original_train_one = protocol._train_one
    original_pcgrad = v22._train_candidate_pcgrad

    def save_state(model: torch.nn.Module, name: str) -> None:
        path = fold_dir / name
        torch.save({"state_dict": {key: value.detach().cpu()
                                   for key, value in model.state_dict().items()}}, path)
        captured[name] = path

    def capture_train_one(*args, **kwargs):
        model, info = original_train_one(*args, **kwargs)
        if args and args[0] == "phaseformer_exact":
            save_state(model, "phaseformer_exact.pt")
        return model, info

    def capture_pcgrad(model, *args, **kwargs):
        info = original_pcgrad(model, *args, **kwargs)
        save_state(model, "scpcr_candidate.pt")
        return info

    protocol._train_one = capture_train_one
    v22._train_candidate_pcgrad = capture_pcgrad
    try:
        result = v22.run(
            target,
            seed,
            output_root=OUTPUT,
            steps=2048,
            device=device,
            torch_threads=torch_threads,
            source_diagnostic=False,
            archive_path=protocol.ARCHIVE,
        )
    finally:
        protocol._train_one = original_train_one
        v22._train_candidate_pcgrad = original_pcgrad

    code_paths = (
        Path(__file__).resolve(),
        protocol.HERE,
        v21.HERE,
        v22.HERE,
        ROOT / "scprc/models/scprc.py",
        ROOT / "scprc/training/relation_attenuation.py",
        ROOT / "scprc/baselines/phaseformer_adapter.py",
    )
    result["current_run_provenance"] = {
        "run_status": "retrospective_retraining",
        "blind_confirmation": False,
        "protocol_id": "scpcr-matched-protocol-20261023",
        "training_budget": {
            "configured_source_minibatches": 2048,
            "batch_windows": 4,
            "effective_source_windows": 8192,
            "optimizer": "AdamW",
            "candidate_gradient_aggregation": "PCGrad across source-city losses",
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path)
                        for path in code_paths if path.is_file()},
        "checkpoint_sha256": {name: sha256(path) for name, path in captured.items()},
        "predictions_sha256": sha256(fold_dir / "predictions.npz"),
        "legacy_import_compatibility": {
            "shim_module": "calibration_protocol_constants",
            "purpose": "satisfy an unused legacy import in the source-certificate helper using the frozen current CHARGED protocol constants",
            "candidate_threshold_and_support_functions": "from current source-certified helper",
        },
        "checkpoint_files": sorted(captured),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=protocol.CITY_CODES)
    parser.add_argument("--seed", required=True, type=int, choices=(13, 17, 29))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    result = run_fold(args.target, args.seed, args.device, args.torch_threads)
    print(json.dumps({"status": result["status"], "target": args.target,
                      "seed": args.seed, "metrics": result["metrics"],
                      "output": str(OUTPUT / f"target_{args.target.lower()}_seed{args.seed}")},
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

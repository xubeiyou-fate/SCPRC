#!/usr/bin/env python3
"""Canonical-PCGrad single-component SCPRC ablation runner.

The runner keeps the frozen CHARGED protocol and changes only the three fixed
relation gates.  Every arm is trained from an independent seed reset and gets
its own checkpoint and prediction archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
EXPERIMENTS = ROOT / 'scprc/training'
PROTOCOL_FILE = EXPERIMENTS / 'scprc_matched_protocol.json'
OUTPUT_DEFAULT = ROOT / 'experiments/independent_gates'
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The source adapter retains a historical import for a module that is not
# needed by the canonical PCGrad path.
legacy_name = "calibration_protocol_constants"
if legacy_name not in sys.modules:
    legacy = types.ModuleType(legacy_name)
    legacy.CONTEXT = 168
    legacy.HORIZON = 24
    legacy.TRAIN_END = 2928
    sys.modules[legacy_name] = legacy

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.train_scprc as pcgrad  # noqa: E402
from scprc.models.scprc import (  # noqa: E402
    SCPRC,
)


ARMS = {
    "exact": (False, False, False),
    "scale_only": (True, False, False),
    "peer_only": (False, True, False),
    "source_disagreement_only": (False, False, True),
    "full": (True, True, True),
}
SEEDS = (13, 17, 29)
PROTOCOL_ID = "scpcr-matched-baseline-20261023"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Reset all RNGs before model construction and before training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(arm: str, n_sites: int, threshold: float) -> torch.nn.Module:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    if arm == "exact":
        return protocol._build_model("phaseformer_exact", n_sites).float()
    scale, peer, source = ARMS[arm]
    return SCPRC(
        channels=n_sites, seq_len=168, pred_len=24, period_len=24,
        latent_dim=32, phase_layers=2, routers=1, heads=8,
        dropout=0.1, pos_embed=True, relation_mode="phase",
        relation_hidden=16, relation_rank=8,
        support_scale=1.0, source_relation_threshold=threshold,
        use_scale_gate=scale, use_peer_gate=peer,
        use_source_disagreement_gate=source,
    ).float()


def _validate_gate_flags(arm: str, model: torch.nn.Module) -> None:
    """Check that the named arm actually selects the intended adapter gates."""
    if arm == "exact":
        return
    expected = {
        "use_scale_gate": ARMS[arm][0],
        "use_peer_gate": ARMS[arm][1],
        "use_source_disagreement_gate": ARMS[arm][2],
    }
    actual = {key: bool(getattr(model, key)) for key in expected}
    if actual != expected:
        raise AssertionError(f"gate configuration mismatch for {arm}: {actual} != {expected}")


def run_fold(target: str, seed: int, *, output_root: Path, steps: int,
             device: str, torch_threads: int) -> dict:
    target = target.upper()
    if target not in protocol.CITY_CODES or seed not in SEEDS:
        raise ValueError("target/seed is outside the frozen canonical matrix")
    if steps < 1 or torch_threads < 1:
        raise ValueError("steps and torch_threads must be positive")
    out = output_root / f"target_{target.lower()}_seed{seed}"
    if (out / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite {out / 'result.json'}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    cities, provenance = protocol.load_cities(protocol.ARCHIVE)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    n_sites = len(cities[target]["site_ids"])
    threshold = pcgrad.v21.cert._estimate_threshold(
        cities, source_ids, mean, std, seed=seed, device=device, quantile=0.90,
    )
    predictions = {
        "label_starts": starts,
        "truth_raw": truth,
        "site_ids": np.asarray(cities[target]["site_ids"], dtype=str),
    }
    training = {}
    checkpoints = {}
    started = time.perf_counter()
    for arm in ARMS:
        # Model construction consumes RNG state.  Reset before construction
        # so each arm follows the same per-arm initialization convention as
        # the canonical matched runner, independent of loop order.
        seed_everything(seed)
        model = build_model(arm, n_sites, float(threshold)).to(device)
        _validate_gate_flags(arm, model)
        info = pcgrad._train_candidate_pcgrad(
            model, cities, source_ids, mean, std, seed=seed, steps=steps,
            device=device, torch_threads=torch_threads,
        )
        prediction = protocol._predict(
            model, target_values, starts, mean, std, device=device, batch_size=4,
        ).astype(np.float32)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid prediction for {arm}: {prediction.shape}")
        predictions[f"pred_{arm}_raw"] = prediction
        training[arm] = {
            **info,
            "arm": arm,
            "gate_flags": {
                "use_scale_gate": ARMS[arm][0],
                "use_peer_gate": ARMS[arm][1],
                "use_source_disagreement_gate": ARMS[arm][2],
            },
            "source_relation_threshold": float(threshold),
            "metrics": protocol._score(prediction, truth),
        }
        checkpoint = out / f"{arm}.pt"
        torch.save({
            "arm": arm,
            "model_name": "phaseformer_exact" if arm == "exact" else "scprc",
            "support_scale": 1.0 if arm != "exact" else None,
            "gate_flags": training[arm]["gate_flags"],
            "source_relation_threshold": float(threshold),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, checkpoint)
        checkpoints[arm] = checkpoint
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "experiment": "canonical-PCGrad independent-gate component ablation rebuild",
        "status": "completed_smoke" if steps != 2048 else "completed",
        "target_city": target, "seed": seed,
        "source_cities": list(source_ids),
        "protocol_file": str(PROTOCOL_FILE),
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "protocol_id": PROTOCOL_ID,
        "retrospective": True, "blind_confirmation": False,
        "input": {"features": ["volume"], "context_hours": 168, "horizon_hours": 24},
        "training_budget": {
            "source_minibatches_total": steps, "batch_size_windows": 4,
            "optimizer": "AdamW", "gradient_aggregation": "canonical PCGrad",
            "loss": "standardized volume MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        },
        "arms": list(ARMS), "training": training,
        "checkpoints": {arm: str(path) for arm, path in checkpoints.items()},
        "predictions": str(out / "predictions.npz"),
        "prediction_sha256": None,
        "data_provenance": provenance,
        "wall_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(out / "predictions.npz", **predictions)
    result["prediction_sha256"] = sha256(out / "predictions.npz")
    (out / "protocol.json").write_text(json.dumps({
        "protocol_id": result["protocol_id"],
        "protocol_sha256": result["protocol_sha256"],
        "gate_definition": {
            arm: {"use_scale_gate": flags[0], "use_peer_gate": flags[1],
                  "use_source_disagreement_gate": flags[2]}
            for arm, flags in ARMS.items()
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def smoke() -> dict:
    """Exercise all gate combinations without reading data or training."""
    seed_everything(20261030)
    x = torch.randn(2, protocol.CONTEXT, 11)
    outputs = {}
    for arm in ARMS:
        seed_everything(20261030)
        model = build_model(arm, 11, threshold=1.0).eval()
        _validate_gate_flags(arm, model)
        with torch.no_grad():
            if arm == "exact":
                y = model(x)
            else:
                y, diagnostics = model(x, return_diagnostics=True)
                required = {"relation_disagreement", "source_peer_disagreement"}
                if not required.issubset(diagnostics):
                    raise AssertionError(f"missing adapter diagnostics for {arm}")
                if not all(torch.isfinite(value).all() for value in diagnostics.values()
                           if torch.is_tensor(value)):
                    raise AssertionError(f"non-finite adapter diagnostics for {arm}")
        expected_shape = (2, protocol.HORIZON, 11)
        if tuple(y.shape) != expected_shape or not torch.isfinite(y).all():
            raise AssertionError(f"invalid smoke output for {arm}: {tuple(y.shape)}")
        outputs[arm] = {
            "shape": list(y.shape),
            "gate_flags": {
                key: bool(getattr(model, key))
                for key in (
                    "use_scale_gate", "use_peer_gate",
                    "use_source_disagreement_gate",
                )
                if hasattr(model, key)
            },
        }
        del model
    return {
        "status": "passed",
        "smoke_only": True,
        "data_read": False,
        "training_started": False,
        "arms": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=protocol.CITY_CODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(smoke(), sort_keys=True))
        return 0
    if args.target is None or args.seed is None:
        parser.error("--target and --seed are required unless --smoke-only is set")
    result = run_fold(args.target, args.seed, output_root=args.output_root,
                      steps=args.steps, device=args.device,
                      torch_threads=args.torch_threads)
    print(json.dumps({"status": result["status"], "target_city": args.target,
                      "seed": args.seed, "arms": result["arms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

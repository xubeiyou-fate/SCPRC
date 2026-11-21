#!/usr/bin/env python3
"""Train the four missing strict relation-adapter controls.

This runner has a separate output root and never loads or overwrites an old
checkpoint. It reuses the frozen protocol and PCGrad training function while
making the gate definition and daily relation mode explicit in each result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OLD_RUNNER = ROOT / 'experiments/independent_gates'
if str(OLD_RUNNER) not in sys.path:
    sys.path.insert(0, str(OLD_RUNNER))
import experiments.independent_gates.run_canonical_component_ablation as base  # noqa: E402


protocol = base.protocol
pcgrad = base.pcgrad
SCPRC = base.SCPRC
SEEDS = (13, 17, 29)
PROTOCOL_ID = "scpcr-matched-baseline-20261023"
OUTPUT_DEFAULT = HERE.parent / "runs"

ARMS = {
    "full_drop_scale": {"gates": (False, True, True), "relation_mode": "phase", "threshold_policy": "source_q90"},
    "full_drop_peer": {"gates": (True, False, True), "relation_mode": "phase", "threshold_policy": "source_q90"},
    "full_drop_source_envelope": {"gates": (True, True, False), "relation_mode": "phase", "threshold_policy": "not_estimated_gate_disabled"},
    "daily_full_attenuation": {"gates": (True, True, True), "relation_mode": "phase_agnostic_daily", "threshold_policy": "source_q90"},
}


def build_model(arm: str, n_sites: int, threshold: float) -> torch.nn.Module:
    spec = ARMS[arm]
    scale, peer, source = spec["gates"]
    return SCPRC(
        channels=n_sites, seq_len=168, pred_len=24, period_len=24,
        latent_dim=32, phase_layers=2, routers=1, heads=8,
        dropout=0.1, pos_embed=True, relation_mode=spec["relation_mode"],
        relation_hidden=16, relation_rank=8, support_scale=1.0,
        source_relation_threshold=threshold, use_scale_gate=scale,
        use_peer_gate=peer, use_source_disagreement_gate=source,
    ).float()


def validate_model(arm: str, model: torch.nn.Module) -> None:
    expected = {
        "use_scale_gate": ARMS[arm]["gates"][0],
        "use_peer_gate": ARMS[arm]["gates"][1],
        "use_source_disagreement_gate": ARMS[arm]["gates"][2],
    }
    actual = {key: bool(getattr(model, key)) for key in expected}
    if actual != expected:
        raise AssertionError(f"{arm}: gate flags {actual} != {expected}")
    if getattr(model.relation, "mode", None) != ARMS[arm]["relation_mode"]:
        raise AssertionError(f"{arm}: relation mode mismatch")


def run_fold(target: str, seed: int, *, output_root: Path, steps: int,
             device: str, torch_threads: int) -> dict:
    target = target.upper()
    if target not in protocol.CITY_CODES or seed not in SEEDS:
        raise ValueError("target/seed is outside the frozen canonical matrix")
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
    threshold = None
    if any(spec["threshold_policy"] == "source_q90" for spec in ARMS.values()):
        threshold = float(pcgrad.v21.cert._estimate_threshold(
            cities, source_ids, mean, std, seed=seed, device=device, quantile=0.90,
        ))
    predictions = {
        "label_starts": starts,
        "truth_raw": truth,
        "site_ids": np.asarray(cities[target]["site_ids"], dtype=str),
    }
    training = {}
    checkpoints = {}
    started = time.perf_counter()
    for arm, spec in ARMS.items():
        base.seed_everything(seed)
        model_threshold = threshold if spec["threshold_policy"] == "source_q90" else 1.0
        model = build_model(arm, n_sites, model_threshold).to(device)
        validate_model(arm, model)
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
        flags = {
            "use_scale_gate": spec["gates"][0],
            "use_peer_gate": spec["gates"][1],
            "use_source_disagreement_gate": spec["gates"][2],
        }
        training[arm] = {
            **info, "arm": arm, "relation_mode": spec["relation_mode"],
            "gate_flags": flags, "threshold_policy": spec["threshold_policy"],
            "source_relation_threshold": threshold if spec["threshold_policy"] == "source_q90" else None,
            "metrics": protocol._score(prediction, truth),
        }
        checkpoint = out / f"{arm}.pt"
        torch.save({
            "arm": arm, "model_name": "scprc",
            "relation_mode": spec["relation_mode"], "support_scale": 1.0,
            "gate_flags": flags, "threshold_policy": spec["threshold_policy"],
            "source_relation_threshold": threshold if spec["threshold_policy"] == "source_q90" else None,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, checkpoint)
        checkpoints[arm] = checkpoint
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "experiment": "strict relation-adapter energy ablation",
        "status": "completed_smoke" if steps != 2048 else "completed",
        "target_city": target, "seed": seed, "source_cities": list(source_ids),
        "protocol_file": str(base.PROTOCOL_FILE), "protocol_sha256": base.sha256(base.PROTOCOL_FILE),
        "protocol_id": PROTOCOL_ID, "retrospective": True, "blind_confirmation": False,
        "input": {"features": ["volume"], "context_hours": 168, "horizon_hours": 24},
        "training_budget": {
            "source_minibatches_total": steps, "batch_size_windows": 4,
            "optimizer": "AdamW", "gradient_aggregation": "canonical PCGrad",
            "loss": "standardized volume MSE + 0.15 multi-resolution consistency (2,3,4,6,12)",
        },
        "arms": list(ARMS), "training": training,
        "checkpoints": {arm: str(path) for arm, path in checkpoints.items()},
        "predictions": str(out / "predictions.npz"),
        "data_provenance": provenance, "wall_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(out / "predictions.npz", **predictions)
    (out / "protocol.json").write_text(json.dumps({
        "protocol_id": PROTOCOL_ID, "protocol_sha256": result["protocol_sha256"],
        "arms": ARMS, "threshold_policy": "source q90 is estimated only for arms that apply it",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def smoke() -> dict:
    base.seed_everything(20261104)
    x = torch.randn(2, protocol.CONTEXT, 11)
    outputs = {}
    for arm in ARMS:
        base.seed_everything(20261104)
        model = build_model(arm, 11, threshold=1.0).eval()
        validate_model(arm, model)
        with torch.no_grad():
            y, diagnostics = model(x, return_diagnostics=True)
        if tuple(y.shape) != (2, protocol.HORIZON, 11) or not torch.isfinite(y).all():
            raise AssertionError(f"invalid smoke output for {arm}: {tuple(y.shape)}")
        outputs[arm] = {"shape": list(y.shape), "relation_mode": ARMS[arm]["relation_mode"],
                        "gate_flags": ARMS[arm]["gates"],
                        "diagnostic_keys": sorted(diagnostics)}
        del model
    return {"status": "passed", "smoke_only": True, "data_read": False,
            "training_started": False, "arms": outputs}


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

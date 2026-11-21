#!/usr/bin/env python3
"""Source-only LOCO for source-certified phase-relation calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scprc.data_adapters.charged.city_adapter import ARCHIVE, CITY_CODES  # noqa: E402
import types
import scprc.training.training_protocol as protocol
# This source snapshot includes only the canonical constants compatibility path.
# Historical residual-calibrator training helpers were not part of the release.
base = types.SimpleNamespace(CONTEXT=protocol.CONTEXT, HORIZON=protocol.HORIZON,
                             TRAIN_END=protocol.TRAIN_END, protocol=protocol)  # noqa: E402
from scprc.models.residual_calibration import (  # noqa: E402
    SourceCertifiedPhaseRelationCalibrator, source_relation_disagreement,
)


def _estimate_threshold(cities, source_ids, mean, std, *, seed: int = 0,
                        device: str = "cuda", samples_per_city: int = 128,
                        quantile: float = 0.90) -> float:
    rng = np.random.default_rng(seed + 7001)
    starts = np.arange(base.CONTEXT, base.TRAIN_END - base.HORIZON + 1, dtype=np.int64)
    values = []
    with torch.no_grad():
        for city in source_ids:
            choices = rng.choice(starts, size=min(samples_per_city, len(starts)), replace=False)
            raw = np.stack([cities[city]["values"][s - base.CONTEXT:s] for s in choices]).astype(np.float32)
            x = torch.from_numpy((raw - mean) / std).to(device)
            values.extend(source_relation_disagreement(x).cpu().tolist())
    if not values:
        raise ValueError("no source windows available for relation threshold")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _profile(values: np.ndarray) -> np.ndarray:
    """History-only city profile used for a fixed source-support certificate."""
    values = values[:base.TRAIN_END].astype(np.float64)
    site_mean = values.mean(axis=0)
    site_std = values.std(axis=0)
    zero_fraction = (values == 0).mean(axis=0)
    return np.asarray([
        np.log1p(site_mean).mean(), np.log1p(site_mean).std(),
        np.log1p(site_std).mean(), np.log1p(site_std).std(),
        zero_fraction.mean(), zero_fraction.std(),
    ], dtype=np.float64)


def _source_support_scale(cities, source_ids: tuple[str, ...], target: str,
                          *, radius_ratio: float = 2.0) -> tuple[float, dict]:
    """Project out corrections for a clearly out-of-support city profile.

    The profile is formed from the observed target prefix and source prefixes;
    no forecast-horizon labels are read.  Distances are standardized using the
    five source-city profiles.  A target farther than twice the furthest source
    distance is outside the source support and receives the exact backbone.
    """
    source_profiles = np.stack([_profile(cities[c]["values"]) for c in source_ids])
    center = source_profiles.mean(axis=0)
    spread = source_profiles.std(axis=0) + 1e-6
    source_distances = np.linalg.norm((source_profiles - center) / spread, axis=1)
    target_distance = float(np.linalg.norm((_profile(cities[target]["values"]) - center) / spread))
    limit = float(radius_ratio * source_distances.max())
    scale = 0.0 if target_distance > limit else 1.0
    return scale, {
        "target_profile_distance": target_distance,
        "source_profile_max_distance": float(source_distances.max()),
        "source_support_radius_ratio": float(radius_ratio),
        "source_support_limit": limit,
        "source_support_passed": bool(scale > 0.0),
    }


def run(target: str, seed: int, *, output_root: Path,
        backbone_steps: int = 2048, calibration_steps: int = 64,
        device: str = "cuda", residual_bound: float = 0.20,
        torch_threads: int = 2, calibration_loss: str = "robust",
        calibrator_hidden: int = 16, early_stop: bool = True,
        calibration_eval_interval: int = 8, tail_rmse_tolerance: float = 0.02,
        source_quantile: float = 0.90, archive_path: Path = ARCHIVE) -> dict:
    target = target.upper()
    if target not in CITY_CODES:
        raise ValueError(f"target must be one of {CITY_CODES}")
    cities, provenance = base.protocol.load_cities(archive_path)
    source_ids = tuple(city for city in CITY_CODES if city != target)
    mean, std = base.protocol.source_scaler(cities, source_ids)
    threshold = _estimate_threshold(
        cities, source_ids, mean, std, seed=seed, device=device,
        quantile=source_quantile,
    )
    support_scale, support_info = _source_support_scale(cities, source_ids, target)
    starts, truth, target_values = base.protocol._validation_data(cities[target])
    backbone, backbone_info = base.protocol._train_one(
        "phaseformer_exact", cities, source_ids,
        len(cities[target]["site_ids"]), mean, std, seed=seed,
        steps=backbone_steps, device=device, torch_threads=torch_threads,
    )
    calibrator = SourceCertifiedPhaseRelationCalibrator(
        horizon=base.HORIZON, residual_bound=residual_bound,
        hidden=calibrator_hidden, trust_threshold=threshold,
        support_scale=support_scale,
    ).float().to(device)
    calibration_info = base._fit_calibrator(
        backbone, calibrator, cities, source_ids, mean, std,
        seed=seed, steps=calibration_steps, device=device,
        torch_threads=torch_threads, calibration_loss=calibration_loss,
        early_stop=early_stop, calibration_eval_interval=calibration_eval_interval,
        tail_rmse_tolerance=tail_rmse_tolerance,
    )
    use_correction = bool(calibration_info["inner_correction_selected"] and support_scale > 0.0)
    baseline = base._apply(backbone, calibrator, target_values, starts, mean, std,
                           device=device, use_correction=False)
    candidate = base._apply(backbone, calibrator, target_values, starts, mean, std,
                            device=device, use_correction=True)
    release = candidate if use_correction else baseline
    predictions = {"phaseformer_exact": baseline,
                   "source_certified_relation_candidate": candidate,
                   "source_certified_relation": release}
    out = output_root / f"target_{target.lower()}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz", label_starts=starts,
        truth_raw=truth, site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
        **{f"pred_{name}_raw": value.astype(np.float32)
           for name, value in predictions.items()},
    )
    result = {
        "experiment": "CHARGED v3 source-only LOCO: source-certified phase-relation calibration",
        "status": "completed", "target_city": target,
        "source_cities": list(source_ids), "seed": seed,
        "backbone": backbone_info, "calibrator": calibration_info,
        "selected_model": "source_certified_relation" if use_correction else "phaseformer_exact",
        "protocol": {
            "context_hours": base.CONTEXT, "horizon_hours": base.HORIZON,
            "source_train_end_exclusive_index": base.TRAIN_END,
            "target_validation_interval": "2023-08-01 through 2023-08-31",
            "validation_origins": int(len(starts)),
            "target_labels_used_for_optimization": False,
            "target_statistics_used_for_scaling": False,
            "source_relation_threshold": float(threshold),
            "source_relation_threshold_quantile": float(source_quantile),
            **support_info,
            "relation_support_projection": "exact-backbone fallback when target history profile exceeds 2x source radius",
            "fixed_residual_bound": float(residual_bound),
            "calibration_loss": calibration_loss,
            "calibrator_hidden": int(calibrator_hidden),
            "early_stop": bool(early_stop),
            "calibration_eval_interval": int(calibration_eval_interval),
            "tail_rmse_tolerance": float(tail_rmse_tolerance),
            "learned_gate": False, "expert_pool": False,
            "relation_in_backbone_input": False,
            "september_values_materialized": False,
        },
        "source_scaler_mean": mean, "source_scaler_std": std,
        "data_provenance": provenance,
        "metrics": {name: base.protocol._score(pred, truth)
                     for name, pred in predictions.items()},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    del backbone, calibrator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CITY_CODES, required=True)
    parser.add_argument("--seed", type=int, choices=(13, 17), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path,
                        default=HERE.parent / "charged_source_certified_relation_loco")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--backbone-steps", type=int, default=2048)
    parser.add_argument("--calibration-steps", type=int, default=64)
    parser.add_argument("--calibrator-hidden", type=int, default=16)
    parser.add_argument("--residual-bound", type=float, default=0.20)
    parser.add_argument("--calibration-loss", choices=("mse", "robust", "mae_robust", "cvar_robust"), default="robust")
    parser.add_argument("--calibration-eval-interval", type=int, default=8)
    parser.add_argument("--tail-rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--source-quantile", type=float, default=0.90)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()
    result = run(
        args.target, args.seed, output_root=args.output_root,
        backbone_steps=args.backbone_steps, calibration_steps=args.calibration_steps,
        device=args.device, residual_bound=args.residual_bound,
        torch_threads=args.torch_threads, calibration_loss=args.calibration_loss,
        calibrator_hidden=args.calibrator_hidden, early_stop=not args.no_early_stop,
        calibration_eval_interval=args.calibration_eval_interval,
        tail_rmse_tolerance=args.tail_rmse_tolerance,
        source_quantile=args.source_quantile, archive_path=args.archive,
    )
    print(json.dumps({"target_city": result["target_city"], "seed": result["seed"],
                      "selected_model": result["selected_model"],
                      "metrics": result["metrics"],
                      "source_relation_threshold": result["protocol"]["source_relation_threshold"],
                      "inner_correction_selected": result["calibrator"]["inner_correction_selected"]},
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

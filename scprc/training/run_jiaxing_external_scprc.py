#!/usr/bin/env python3
"""One-shot, source-only CHARGED-to-Jiaxing external-city evaluation.

The data/protocol are locked by prepare_jiaxing_scprc_external.py.
The only models trained are source-only CHARGED models. Jiaxing labels enter
only the final forecasting score after the models and certificate are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scprc.training.training_protocol as phase_protocol  # noqa: E402
import scprc.training.train_scprc as v22  # noqa: E402
import scprc.training.source_calibration_support as cert  # noqa: E402
import scprc.training.scprc_training_support as v21  # noqa: E402
import scprc.training.run_charged_covariate_hptn_phaseformer_loco as patch_protocol  # noqa: E402
import scprc.training.run_charged_patchtst_2var_loco as patch_runner  # noqa: E402
from scprc.data_adapters.charged.city_adapter import CITY_CODES  # noqa: E402

EXPERIMENTS = ROOT / 'scprc/training'
BASE = EXPERIMENTS / "jiaxing_scpcr_external_20261021"
FREEZE = BASE / "protocol_freeze.json"
PREPARED = BASE / "jiaxing_hourly_panel.npz"
CONTEXT = 168
HORIZON = 24
STEPS = 2048
BATCH_SIZE = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel() -> dict[str, Any]:
    if not FREEZE.is_file() or not PREPARED.is_file():
        raise FileNotFoundError("run prepare_jiaxing_scprc_external.py first")
    frozen = json.loads(FREEZE.read_text())
    if frozen["prepared_sha256"] != sha256(PREPARED):
        raise ValueError("prepared panel hash differs from the frozen manifest")
    z = np.load(PREPARED, allow_pickle=False)
    panel = {k: z[k] for k in z.files}
    values = panel["energy_kwh"].astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 13 or len(panel["site_ids"]) != 13:
        raise ValueError(f"unexpected panel shape {values.shape}")
    duration = panel["duration_hours"].astype(np.float32)
    if duration.shape != values.shape or not np.isfinite(values).all() or not np.isfinite(duration).all():
        raise ValueError("nonfinite or misaligned external values")
    if (values < 0).any() or (duration < 0).any():
        raise ValueError("negative external load or duration")
    return {**panel, "values": values, "duration": duration, "freeze": frozen}


def starts_for(panel: dict[str, Any]) -> np.ndarray:
    return panel["test_starts"].astype(np.int64)


def predict_univariate(model: torch.nn.Module, values: np.ndarray,
                       starts: np.ndarray, mean: float, std: float,
                       device: str, batch_size: int = 16) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            raw = np.stack([values[s - CONTEXT:s] for s in batch_starts]).astype(np.float32)
            x = torch.from_numpy((raw - mean) / std).to(device)
            pred = model(x)
            if pred.ndim != 3 or pred.shape[1] != HORIZON:
                raise RuntimeError(f"unexpected forecast shape {tuple(pred.shape)}")
            result.append(pred.detach().cpu().numpy() * std + mean)
    return np.concatenate(result, axis=0).astype(np.float32)


def predict_patchtst(model, values: np.ndarray, starts: np.ndarray,
                     mean: np.ndarray, std: np.ndarray, device: str,
                     batch_size: int = 8) -> np.ndarray:
    result = []
    model.eval()
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        xs = np.stack([values[s - CONTEXT:s] for s in batch_starts]).astype(np.float32)
        # [B,T,N,V] -> [B*N,T,V], matching the source PatchTST harness.
        xs = (xs - mean.reshape(1, 1, 1, 2)) / std.reshape(1, 1, 1, 2)
        batch, _, nodes, _ = xs.shape
        node_x = torch.from_numpy(xs).permute(0, 2, 1, 3).contiguous().reshape(
            batch * nodes, CONTEXT, 2
        ).to(device)
        with torch.no_grad():
            pred_scaled = model(node_x).reshape(batch, nodes, HORIZON).transpose(1, 2)
        result.append((pred_scaled.detach().cpu().numpy() * std[0] + mean[0]).astype(np.float32))
    return np.concatenate(result, axis=0)


def naive_predictions(values: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    daily, weekly, persistence = [], [], []
    for s in starts:
        history = values[int(s) - CONTEXT:int(s)]
        daily.append(np.repeat(history[-24:].mean(axis=0, keepdims=True), HORIZON, axis=0))
        weekly.append(history[-168:-144])
        persistence.append(np.repeat(history[-1:], HORIZON, axis=0))
    return {"persistence": np.asarray(persistence, dtype=np.float32),
            "daily_mean": np.asarray(daily, dtype=np.float32),
            "weekly_naive": np.asarray(weekly, dtype=np.float32)}


def score(pred: np.ndarray, truth: np.ndarray, prefix_std: float) -> dict[str, Any]:
    err = pred.astype(np.float64) - truth.astype(np.float64)
    site_rmse = np.sqrt(np.mean(err * err, axis=(0, 1)))
    site_mae = np.mean(np.abs(err), axis=(0, 1))
    total_err = pred.sum(axis=2, dtype=np.float64) - truth.sum(axis=2, dtype=np.float64)
    window_rmse = np.sqrt(np.mean(err * err, axis=(1, 2)))
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "macro_location_rmse": float(site_rmse.mean()),
        "macro_location_mae": float(site_mae.mean()),
        "total_load_rmse": float(np.sqrt(np.mean(total_err * total_err))),
        "total_load_mae": float(np.mean(np.abs(total_err))),
        "nrmse_by_unlabeled_prefix_std": float(np.sqrt(np.mean(err * err)) / max(prefix_std, 1e-8)),
        "window_rmse_p95": float(np.quantile(window_rmse, 0.95)),
        "origins": int(pred.shape[0]), "horizon": int(pred.shape[1]),
        "locations": int(pred.shape[2]),
        "per_location_rmse": site_rmse.tolist(),
        "per_location_mae": site_mae.tolist(),
    }


def run(seed: int, *, output_root: Path, device: str = "cuda",
        torch_threads: int = 1, steps: int = STEPS) -> dict[str, Any]:
    if seed not in (13, 17, 29):
        raise ValueError("locked external seeds are 13, 17, and 29")
    out = output_root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    panel = load_panel()
    values = panel["values"]
    starts = starts_for(panel)
    prefix_end = int(panel["target_prefix_end"])
    test_end = int(panel["test_end"])
    if len(starts) != 2185 or starts[0] != prefix_end or starts[-1] + HORIZON != test_end:
        raise ValueError("test boundary mismatch")
    truth = np.stack([values[int(s):int(s) + HORIZON] for s in starts]).astype(np.float32)

    # The external target is added only to the support-profile calculator. It is
    # never added to the source model-training dictionary.
    source_cities, source_provenance = phase_protocol.load_cities(phase_protocol.ARCHIVE)
    source_ids = tuple(CITY_CODES)
    mean, std = phase_protocol.source_scaler(source_cities, source_ids)
    threshold = cert._estimate_threshold(
        source_cities, source_ids, mean, std, seed=seed,
        device=device, quantile=0.90,
    )
    support_cities = dict(source_cities)
    support_cities["JIAXING"] = {"values": values[:prefix_end]}
    support_scale, support_info = cert._source_support_scale(
        support_cities, source_ids, "JIAXING", radius_ratio=2.0,
    )

    started = time.perf_counter()
    backbone, backbone_info = phase_protocol._train_one(
        "phaseformer_exact", source_cities, source_ids, values.shape[1],
        mean, std, seed=seed, steps=steps, device=device,
        torch_threads=torch_threads,
    )
    # Candidate is always trained for diagnosis. Its strict-release status is
    # selected only by the fixed source-support certificate, never test scores.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    candidate = v21._build_model(values.shape[1], threshold, support_scale)
    candidate_info = v22._train_candidate_pcgrad(
        candidate, source_cities, source_ids, mean, std, seed=seed,
        steps=steps, device=device, torch_threads=torch_threads,
    )
    elapsed_phase_scpcr = time.perf_counter() - started
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in backbone.state_dict().items()}},
               out / "phaseformer_source_state.pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in candidate.state_dict().items()}},
               out / "scpcr_source_state.pt")
    exact_pred = predict_univariate(backbone, values, starts, mean, std, device)
    candidate_pred = predict_univariate(candidate, values, starts, mean, std, device)
    release_name = "scpcr_v22_pcgrad_strict_release" if support_scale > 0 else "phaseformer_exact_fallback"
    release_pred = candidate_pred if support_scale > 0 else exact_pred
    predictions: dict[str, np.ndarray] = {
        "phaseformer_exact": exact_pred,
        "scpcr_v22_pcgrad_candidate_diagnostic": candidate_pred,
        "scpcr_v22_pcgrad_strict_release": release_pred,
    }
    del backbone, candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # External strong time-series baseline. It receives the same two derived
    # historical variables as local source-only PatchTST; all weights are
    # trained on CHARGED source cities only.
    patch_cities, patch_provenance = patch_protocol.load_feature_arrays()
    patch_ids = tuple(CITY_CODES)
    patch_mean, patch_std = patch_protocol.source_only_scalers(patch_cities, patch_ids)
    patch_model, patch_info = patch_runner._fit_model(
        patch_cities, patch_ids, patch_mean, patch_std, seed=seed, steps=steps,
        batch_size=patch_runner.DEFAULT_BATCH_SIZE,
        learning_rate=patch_runner.DEFAULT_LR, device=device,
        torch_threads=torch_threads,
    )
    patch_values = np.stack([values, panel["duration"]], axis=-1).astype(np.float32)
    predictions["patchtst_two_variable_source_only"] = predict_patchtst(
        patch_model, patch_values, starts, patch_mean, patch_std, device,
    )
    torch.save({k: v.detach().cpu() for k, v in patch_model.state_dict().items()},
               out / "patchtst_source_state.pt")
    del patch_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions.update(naive_predictions(values, starts))
    metrics = {name: score(pred, truth, float(values[:prefix_end].std()))
               for name, pred in predictions.items()}
    target_truth = truth  # read only after source models and release rule are fixed
    np.savez_compressed(
        out / "predictions.npz",
        starts=starts,
        label_timestamps=np.asarray([panel["times"][int(s)] for s in starts], dtype=str),
        site_ids=panel["site_ids"],
        truth_kwh=target_truth,
        **{f"pred_{name}": pred.astype(np.float32) for name, pred in predictions.items()},
    )
    result = {
        "status": "completed",
        "experiment": "Frozen CHARGED-to-Jiaxing independent-city SCPRC-v2.2-PCGrad test",
        "city": "Jiaxing, China; transaction data from 13 charging locations in three districts",
        "seed": seed, "steps_per_model": steps,
        "sources": list(source_ids),
        "target_labels_used_for_training_or_thresholds": False,
        "target_scaling": "not used; fixed CHARGED source-only scaler for PhaseFormer/SCPRC; source-only variable scaler for PatchTST",
        "release_rule": "candidate iff fixed source-support certificate passes; otherwise exact PhaseFormer; no target gate search or expert pool",
        "release_selected": release_name,
        "support": support_info,
        "source_relation_threshold_q90": threshold,
        "phaseformer_training": backbone_info,
        "scpcr_training": candidate_info,
        "phaseformer_and_scpcr_training_elapsed_seconds": elapsed_phase_scpcr,
        "patchtst_training": patch_info,
        "metrics": metrics,
        "protocol_freeze_sha256": sha256(FREEZE),
        "prepared_panel_sha256": sha256(PREPARED),
        "source_data_provenance": source_provenance,
        "patchtst_source_data_provenance": patch_provenance,
        "output_predictions": str((out / "predictions.npz").relative_to(ROOT)),
        "method_limitations": [
            "Hourly energy is transaction-derived, uniformly allocated over session elapsed time, not directly metered hourly load.",
            "The public sample contains 13 station locations in one city; one city is insufficient to establish generalization to most cities.",
            "Rolling forecast origins overlap; inference must use paired contiguous-block resampling and condition on three fixed seeds.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(13, 17, 29))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output-root", type=Path, default=BASE / "runs")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        p = load_panel()
        assert len(starts_for(p)) == 2185
        assert p["values"].shape == (15336, 13)
        assert int(p["target_prefix_end"]) == 13128
        print(json.dumps({"status": "passed", "shape": list(p["values"].shape),
                          "origins": len(starts_for(p)), "sites": len(p["site_ids"])}))
        return 0
    result = run(args.seed, output_root=args.output_root, device=args.device,
                 torch_threads=args.torch_threads, steps=args.steps)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "release_selected": result["release_selected"],
                      "support": result["support"],
                      "metrics": result["metrics"]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen-protocol PhaseFormer + DeepSets adapter LOCO matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / 'scprc/training'
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scprc.training.training_protocol as protocol  # noqa: E402
import scprc.training.run_charged_relation_controls as matched_runner  # noqa: E402
from scprc.models.phaseformer_deepsets_adapter import (  # noqa: E402
    PhaseFormerDeepSetsAdapter,
    PhaseWiseDeepSetsRelationAdapter,
    trainable_parameter_count,
)

FROZEN_PROTOCOL = EXPERIMENTS / 'scprc_matched_protocol.json'
EXPECTED_PROTOCOL_SHA256 = "dc0cb75e9622b3d92698cfaa69e956e46d00a39f92bf8b848121979c1fd362d5"
MODEL_PATH = ROOT / "scprc/models/phaseformer_deepsets_adapter.py"
RUNNER_PATH = Path(__file__).resolve()
SEEDS = (13, 17, 29)
TARGETS = tuple(protocol.CITY_CODES)
ADAPTER_TARGET_PARAMETERS = 903
ADAPTER_PARAMETER_COUNT = 895
RUNS = OUT / "runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def protocol_check() -> dict[str, Any]:
    digest = sha256(FROZEN_PROTOCOL)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol hash mismatch: {digest}")
    frozen = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    if frozen["training_budget"]["source_minibatches_total"] != 2048:
        raise RuntimeError("unexpected frozen source minibatch budget")
    if tuple(frozen["training_budget"]["seeds"]) != SEEDS:
        raise RuntimeError("unexpected frozen seed set")
    if tuple(frozen["data_split"]["cities"]) != TARGETS:
        raise RuntimeError("unexpected frozen city set")
    return {"protocol": frozen, "sha256": digest}


def build_model(site_count: int) -> PhaseFormerDeepSetsAdapter:
    return PhaseFormerDeepSetsAdapter(
        channels=site_count,
        seq_len=protocol.CONTEXT,
        pred_len=protocol.HORIZON,
        period_len=24,
        latent_dim=32,
        phase_layers=2,
        routers=1,
        heads=8,
        dropout=0.1,
        pos_embed=True,
        residual_bound=0.2,
    )


def smoke_test() -> dict[str, Any]:
    torch.manual_seed(20261024)
    adapter = PhaseWiseDeepSetsRelationAdapter()
    parameters = trainable_parameter_count(adapter)
    if parameters != ADAPTER_PARAMETER_COUNT:
        raise AssertionError(f"adapter has {parameters} parameters, expected 895")
    if abs(parameters - ADAPTER_TARGET_PARAMETERS) / ADAPTER_TARGET_PARAMETERS > 0.01:
        raise AssertionError("DeepSets adapter is outside the frozen ±1% match")

    permutation_checks = []
    for stations in (1, 5, 11):
        source = torch.randn(2, stations, 24, 7, requires_grad=True)
        output, _ = adapter(source)
        if output.shape != source.shape or not torch.equal(output, source):
            raise AssertionError("zero-initialized correction must be exact identity")
        permutation = torch.randperm(stations)
        permuted, _ = adapter(source.detach()[:, permutation])
        if not torch.allclose(permuted, output.detach()[:, permutation],
                              atol=1e-6, rtol=1e-6):
            raise AssertionError("adapter is not permutation equivariant")
        (output.square().mean()).backward()
        if any(parameter.grad is not None and
               not torch.isfinite(parameter.grad).all()
               for parameter in adapter.parameters()):
            raise AssertionError("adapter produced non-finite gradients")
        permutation_checks.append(stations)

    torch.manual_seed(13)
    model = build_model(site_count=5).eval()
    history = torch.randn(1, protocol.CONTEXT, 5)
    with torch.no_grad():
        forecast, diagnostics = model(history, return_diagnostics=True)
    if forecast.shape != (1, protocol.HORIZON, 5):
        raise AssertionError(f"unexpected PhaseFormer output shape {forecast.shape}")
    if not torch.isfinite(forecast).all() or not torch.isfinite(
        diagnostics["correction_abs_max"]
    ):
        raise AssertionError("PhaseFormer integration returned non-finite values")
    total_parameters = trainable_parameter_count(model)
    expected_total = 36_360 + parameters
    if total_parameters != expected_total:
        raise AssertionError(
            f"unexpected total params {total_parameters}, expected {expected_total}"
        )
    return {
        "status": "passed",
        "device": "cpu",
        "adapter_parameters": parameters,
        "adapter_target_parameters": ADAPTER_TARGET_PARAMETERS,
        "adapter_relative_difference": (parameters - ADAPTER_TARGET_PARAMETERS)
        / ADAPTER_TARGET_PARAMETERS,
        "total_model_parameters": total_parameters,
        "expected_total_model_parameters": expected_total,
        "permutation_equivariance_station_counts": permutation_checks,
        "identity_initialization": True,
        "finite_gradients": True,
        "phaseformer_output_shape": list(forecast.shape),
    }


def environment_record(device: str, torch_threads: int) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        "device": device,
        "torch_threads": torch_threads,
        "protocol_sha256": sha256(FROZEN_PROTOCOL),
        "model_sha256": sha256(MODEL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "matched_pcgrad_runner_sha256": sha256(
            EXPERIMENTS / "run_charged_relation_controls.py"
        ),
    }


def run_fold(target: str, seed: int, cities: dict[str, dict[str, Any]],
             provenance: dict[str, Any], *, device: str, torch_threads: int,
             attempt: int) -> dict[str, Any]:
    fold = RUNS / f"target_{target.lower()}_seed{seed}" / f"attempt_{attempt:02d}"
    fold.mkdir(parents=True, exist_ok=False)
    source_ids = tuple(city for city in protocol.CITY_CODES if city != target)
    mean, std = protocol.source_scaler(cities, source_ids)
    starts, truth, target_values = protocol._validation_data(cities[target])
    if len(starts) != 721:
        raise ValueError(f"expected 721 August origins, got {len(starts)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(len(cities[target]["site_ids"]))
    total_parameters = trainable_parameter_count(model)
    adapter_parameters = trainable_parameter_count(model.relation)
    if adapter_parameters != ADAPTER_PARAMETER_COUNT:
        raise RuntimeError(f"parameter-count mismatch: {adapter_parameters}")

    training = matched_runner._train_pcgrad(
        model,
        cities,
        source_ids,
        mean,
        std,
        seed=seed,
        steps=2048,
        device=device,
        torch_threads=torch_threads,
    )
    if training["effective_source_windows"] != 8192 or training["optimizer_steps"] != 410:
        raise RuntimeError("PCGrad runner did not meet frozen exposure/update budget")
    training["adapter_parameters"] = adapter_parameters
    model.eval()
    prediction = matched_runner._predict(
        model, target_values, starts, mean, std, device=device
    )
    if prediction.shape != truth.shape:
        raise ValueError(f"forecast shape {prediction.shape} != truth shape {truth.shape}")
    metrics = matched_runner._metric(prediction, truth)

    checkpoint = fold / "trained_checkpoint.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": "PhaseFormer_DeepSets_relation_adapter",
        "model_scope": "PhaseFormer exact backbone with only its input relation adapter swapped",
        "target_city": target,
        "seed": seed,
        "source_scaler_mean": mean,
        "source_scaler_std": std,
        "model_config": {
            "backbone": "PhaseFormer exact",
            "adapter": "phase-wise shared DeepSets mean-pooling",
            "adapter_parameters": adapter_parameters,
            "total_parameters": total_parameters,
            "residual_bound": 0.2,
        },
        "protocol_id": "scpcr-matched-baseline-20261023",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "model_sha256": sha256(MODEL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
    }, checkpoint)
    predictions_path = fold / "predictions.npz"
    with predictions_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            label_starts=starts,
            label_timestamps=np.asarray([
                str(cities[target]["times"][int(start)]) for start in starts
            ]),
            site_ids=np.asarray(cities[target]["site_ids"], dtype=str),
            truth_raw=truth.astype(np.float32),
            prediction_phaseformer_deepsets_adapter=prediction.astype(np.float32),
        )

    result = {
        "status": "completed",
        "experiment": "Frozen-protocol PhaseFormer relation-adapter swap; retrospective August LOCO",
        "model": "PhaseFormer_DeepSets_relation_adapter",
        "model_scope": "same PhaseFormer backbone; only input relation operator replaced",
        "standalone_deepsets_result_reused": False,
        "target_city": target,
        "source_cities": list(source_ids),
        "seed": seed,
        "metrics": metrics,
        "training": training,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
        },
        "protocol": {
            "protocol_id": "scpcr-matched-baseline-20261023",
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "input": "CHARGED v3 volume-only",
            "context_hours": 168,
            "horizon_hours": 24,
            "source_minibatches": 2048,
            "effective_source_windows": 8192,
            "pcgrad_optimizer_updates": 410,
            "evaluation_origins": len(starts),
            "august_role": "retrospective; not blind confirmation",
            "data_provenance": provenance,
        },
    }
    atomic_json(fold / "result.json", result)
    del model
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def next_attempt(target: str, seed: int) -> int:
    parent = RUNS / f"target_{target.lower()}_seed{seed}"
    attempts = [int(path.name.split("_")[-1]) for path in parent.glob("attempt_*")
                if path.is_dir() and path.name.split("_")[-1].isdigit()] if parent.exists() else []
    return max(attempts, default=0) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="run CPU implementation/integration checks only")
    parser.add_argument("--full", action="store_true",
                        help="run all 18 frozen-protocol folds sequentially")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.smoke == args.full:
        parser.error("choose exactly one of --smoke or --full")
    frozen = protocol_check()
    if args.smoke:
        report = smoke_test()
        report["protocol_sha256"] = frozen["sha256"]
        atomic_json(OUT / "smoke_report.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    RUNS.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "protocol.json", frozen["protocol"])
    atomic_json(OUT / "environment.json", environment_record(args.device, args.torch_threads))
    (OUT / "command.txt").write_text(
        f"python {RUNNER_PATH} --full --device {args.device} --torch-threads {args.torch_threads}\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "loading_frozen_dataset", "device": args.device,
                      "targets": TARGETS, "seeds": SEEDS}, separators=(",", ":")),
          flush=True)
    cities, provenance = protocol.load_cities()
    failures = []
    completed = 0
    for target in TARGETS:
        for seed in SEEDS:
            attempt = next_attempt(target, seed)
            try:
                result = run_fold(
                    target, seed, cities, provenance, device=args.device,
                    torch_threads=args.torch_threads, attempt=attempt,
                )
                completed += int(result["status"] == "completed")
                print(json.dumps({"event": "fold_completed", "target": target,
                                  "seed": seed, "metrics": result["metrics"],
                                  "training_seconds": result["training"]["training_seconds"]},
                                 separators=(",", ":")), flush=True)
            except Exception as error:
                failure = {
                    "target_city": target,
                    "seed": seed,
                    "attempt": attempt,
                    "error": repr(error),
                    "time_unix": time.time(),
                }
                failures.append(failure)
                atomic_json(RUNS / f"target_{target.lower()}_seed{seed}"
                            / f"attempt_{attempt:02d}" / "failure.json", failure)
                with (OUT / "failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure, sort_keys=True) + "\n")
                print(json.dumps({"event": "fold_failed", **failure},
                                 separators=(",", ":")), flush=True)
    matrix = {
        "status": "completed" if completed == 18 and not failures else "incomplete",
        "expected_folds": 18,
        "completed_folds_this_invocation": completed,
        "failed_folds_this_invocation": len(failures),
        "targets": TARGETS,
        "seeds": SEEDS,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "model_sha256": sha256(MODEL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "failures": failures,
    }
    atomic_json(OUT / "matrix_execution.json", matrix)
    return 0 if not failures and completed == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())

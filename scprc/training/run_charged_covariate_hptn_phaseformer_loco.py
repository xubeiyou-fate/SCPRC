#!/usr/bin/env python3
"""Matched CHARGED v3 LOCO experiment: volume-duration HPTN vs PhaseFormer.

The runner is deliberately fail-closed: without ``--run`` it performs no data
loading or training. Both models receive the same two historical variables
and share the same PhaseFormer backbone settings. HPTN adds only variable-axis
attention and a zero-mean node transport correction. Training is source-only;
the held-out city is used only for final validation scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
REPRO = ROOT / "scprc"
EXTERNAL = ROOT / 'scprc/data_adapters'
CHARGED = EXTERNAL / "charged"
for _path in (REPRO, EXTERNAL, CHARGED):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scprc.data_adapters.charged.city_adapter import (  # noqa: E402
    ARCHIVE, ARCHIVE_MD5, ARTICLE_DOI, CITY_CODES, DATASET_DOI, LICENSE,
    FIRST_HOUR,
)
from scprc.data_adapters.cross_city_adapter import CONTEXT, HORIZON  # noqa: E402
from scprc.baselines.phaseformer_adapter import (  # noqa: E402
    UPSTREAM_COMMIT as PHASEFORMER_COMMIT,
    PhaseFormerUrbanEV,
)

TRAIN_END = 2928  # 2023-08-01 00:00, exclusive; source labels end Jul 31
VAL_END = 3672    # 2023-09-01 00:00, exclusive; held-out validation is August
VARIABLES = ("volume", "duration")
MODELS = ("hptn", "phaseformer")
SEED_DEFAULTS = (13, 17)


class MatchedPhaseFormer(nn.Module):
    """Shared PhaseFormer run independently for each node (node-as-batch)."""

    def __init__(self, *, latent_dim: int = 32, phase_layers: int = 2,
                 heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.backbone = PhaseFormerUrbanEV(
            seq_len=CONTEXT, pred_len=HORIZON, channels=len(VARIABLES),
            latent_dim=latent_dim, phase_layers=phase_layers, routers=1,
            heads=heads, dropout=dropout, pos_embed=True,
        )

    @staticmethod
    def _node_batch(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 4 or x.shape[2] != len(VARIABLES):
            raise ValueError(f"expected [windows,{CONTEXT},{len(VARIABLES)},nodes] input")
        batch, steps, variables, nodes = x.shape
        if steps != CONTEXT:
            raise ValueError(f"expected {CONTEXT} context steps")
        # Each site is one sample; all sites and cities share the same weights.
        node_x = x.permute(0, 3, 1, 2).contiguous().reshape(
            batch * nodes, steps, variables
        )
        return node_x, batch, nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_x, batch, nodes = self._node_batch(x)
        forecast, _, _ = self.backbone.network(node_x)
        # The official PhaseFormer is channel-independent. It receives both
        # variables, while the matched target decoder selects volume (channel 0).
        volume = forecast[..., 0]
        return volume.reshape(batch, nodes, HORIZON).transpose(1, 2).contiguous()


class CovariateHPTN(MatchedPhaseFormer):
    """PhaseFormer plus cross-variable attention and zero-mean node transport."""

    def __init__(self, *, latent_dim: int = 32, phase_layers: int = 2,
                 heads: int = 8, dropout: float = 0.1, hidden: int = 32):
        super().__init__(latent_dim=latent_dim, phase_layers=phase_layers,
                         heads=heads, dropout=dropout)
        self.cross_variable_heads = max(1, min(4, latent_dim // 8))
        self.cross_variable_attention = nn.MultiheadAttention(
            latent_dim, num_heads=self.cross_variable_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_variable_norm = nn.LayerNorm(latent_dim)
        self.cross_variable_ffn = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim), nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.cross_variable_ffn_norm = nn.LayerNorm(latent_dim)
        self.transport = nn.Sequential(
            nn.Linear(len(VARIABLES) * latent_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, 1),
        )
        # Start as the same volume PhaseFormer forecast. This is a fixed model
        # initialization, not a learned gate or a post-hoc model mixture.
        nn.init.zeros_(self.transport[-1].weight)
        nn.init.zeros_(self.transport[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_x, batch, nodes = self._node_batch(x)
        forecast, latent, _ = self.backbone.network(node_x)
        n_batch = batch * nodes
        # PhaseFormer latent: [windows*nodes, variables, horizon, latent_dim].
        variable_tokens = latent.permute(0, 2, 1, 3).contiguous().reshape(
            n_batch * HORIZON, len(VARIABLES), latent.shape[-1]
        )
        attended_chunks = []
        for start in range(0, variable_tokens.shape[0], 8192):
            part = variable_tokens[start:start + 8192]
            attended, _ = self.cross_variable_attention(
                part, part, part, need_weights=False
            )
            attended_chunks.append(attended)
        attended = torch.cat(attended_chunks, dim=0)
        mixed = self.cross_variable_norm(variable_tokens + attended)
        mixed = self.cross_variable_ffn_norm(mixed + self.cross_variable_ffn(mixed))
        mixed = mixed.reshape(n_batch, HORIZON, len(VARIABLES), latent.shape[-1])
        correction = self.transport(mixed.reshape(n_batch, HORIZON, -1)).squeeze(-1)

        base_volume = forecast[..., 0].reshape(batch, nodes, HORIZON)
        correction = correction.reshape(batch, nodes, HORIZON)
        # Exact zero-sum spatial correction at every horizon/window. The city
        # aggregate remains the shared PhaseFormer volume coordinate.
        correction = correction - correction.mean(dim=1, keepdim=True)
        return (base_volume + correction).transpose(1, 2).contiguous()


def _member_prefix_sha256(archive: zipfile.ZipFile, member: str, data_rows: int) -> str:
    """Hash exactly the header and prefix rows used; leave later labels untouched."""
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        for _ in range(data_rows + 1):  # header plus requested CSV data rows
            line = stream.readline()
            if not line:
                raise ValueError(f"{member} ended before its declared prefix")
            digest.update(line)
    return digest.hexdigest()


def _read_matrix_prefix(archive: zipfile.ZipFile, city: str, variable: str) -> tuple[pd.DatetimeIndex, tuple[str, ...], np.ndarray]:
    """Parse only Apr 1--Aug 31 rows; never materialize September test labels."""
    member = f"{city}/{variable}.csv"
    with archive.open(member) as stream:
        frame = pd.read_csv(stream, nrows=VAL_END)
    if frame.shape[0] != VAL_END or frame.shape[1] < 2:
        raise ValueError(f"{member} does not contain the required {VAL_END}-row prefix")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range(FIRST_HOUR, periods=VAL_END, freq="h")
    node_ids = tuple(str(value) for value in frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not times.equals(expected):
        raise ValueError(f"{member} timestamps differ from the pinned hourly prefix")
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError(f"{member} has empty or duplicate site IDs")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{member} has invalid values")
    return times, node_ids, values


def load_feature_arrays() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load aligned [time,node,2] prefixes and hash the exact source members."""
    cities: dict[str, dict[str, Any]] = {}
    members: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for city in CITY_CODES:
            arrays: dict[str, np.ndarray] = {}
            expected_times = None
            expected_nodes = None
            for variable in VARIABLES:
                times, node_ids, values = _read_matrix_prefix(archive, city, variable)
                if expected_times is None:
                    expected_times, expected_nodes = times, node_ids
                elif not times.equals(expected_times) or node_ids != expected_nodes:
                    raise ValueError(f"{city}/{variable}.csv is not aligned with the other inputs")
                arrays[variable] = values
            values = np.stack([arrays[name] for name in VARIABLES], axis=-1)
            if not np.isfinite(values).all():
                raise ValueError(f"{city} has nonfinite covariates")
            cities[city] = {
                "times": expected_times,
                "node_ids": expected_nodes,
                "values": values,
                "source": {name: f"Zenodo:{DATASET_DOI}:{city}/{name}.csv" for name in VARIABLES},
            }
            members[city] = {
                variable: _member_prefix_sha256(archive, f"{city}/{variable}.csv", VAL_END)
                for variable in VARIABLES
            }
    provenance = {
        "dataset": "CHARGED v3 standard hourly archive; remove_zero=False",
        "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "archive_path": str(ARCHIVE),
        "archive_md5_published": ARCHIVE_MD5,
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "consumed_prefix_sha256_per_city_variable": members,
        "rows_materialized_for_model": VAL_END,
        "model_data_last_index_exclusive": VAL_END,
        "september_test_values_materialized": False,
        "note": "member hashes cover the consumed row prefixes; numeric arrays parsed only through Aug 31",
    }
    return cities, provenance

def _file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_only_scalers(cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Equal-city-weight global mean/std fitted on source prefixes only."""
    city_means, city_second_moments = [], []
    for city in source_ids:
        prefix = cities[city]["values"][:TRAIN_END].astype(np.float64)
        city_means.append(prefix.mean(axis=(0, 1)))
        city_second_moments.append(np.square(prefix).mean(axis=(0, 1)))
    mean = np.mean(np.stack(city_means), axis=0)
    second = np.mean(np.stack(city_second_moments), axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def window_starts(label_start: int, label_end: int) -> np.ndarray:
    """Starts are label indices; every full horizon stays inside its split."""
    first = max(int(label_start), CONTEXT)
    last = int(label_end) - HORIZON
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def make_batch(values: np.ndarray, starts: np.ndarray, mean: np.ndarray,
               std: np.ndarray, *, rng: np.random.Generator,
               batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    picks = rng.choice(starts, size=batch_size, replace=True)
    # Raw city arrays are [time, node, variable], while both model paths use
    # [window, time, variable, node] (node-as-batch).  Transpose before
    # applying the per-variable scaler so the variable axis is unambiguous.
    xs = np.stack([values[s - CONTEXT:s] for s in picks]).astype(np.float32)
    xs = xs.transpose(0, 1, 3, 2)
    ys = np.stack([values[s:s + HORIZON, :, 0] for s in picks]).astype(np.float32)
    xs = (xs - mean.reshape(1, 1, len(VARIABLES), 1)) / std.reshape(1, 1, len(VARIABLES), 1)
    ys = (ys - mean[0]) / std[0]
    return torch.from_numpy(xs), torch.from_numpy(ys)


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def build_model(name: str) -> nn.Module:
    if name == "hptn":
        return CovariateHPTN()
    if name == "phaseformer":
        return MatchedPhaseFormer()
    raise ValueError(f"unknown model: {name}")


def _record_gradient_activity(model: nn.Module, connected: dict[str, bool],
                              nonzero: dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            connected[name] = True
            if bool(torch.count_nonzero(parameter.grad.detach()).item()):
                nonzero[name] = True


def fit_model(name: str, cities: dict[str, dict[str, Any]], source_ids: tuple[str, ...],
              mean: np.ndarray, std: np.ndarray, *, seed: int, steps: int = 512,
              batch_size: int = 4, learning_rate: float = 1e-3,
              device: str = "cuda") -> tuple[nn.Module, dict[str, Any]]:
    """Fit exactly ``steps`` source-only updates; validation is never used here."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(name).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_starts = window_starts(0, TRAIN_END)
    rng = np.random.default_rng(seed)
    updates_by_city = {city: 0 for city in source_ids}
    connected_seen = {parameter_name: False for parameter_name, parameter in model.named_parameters()
                      if parameter.requires_grad}
    nonzero_seen = dict.fromkeys(connected_seen, False)

    for update in range(steps):
        city = source_ids[update % len(source_ids)]
        x, y = make_batch(cities[city]["values"], train_starts, mean, std,
                          rng=rng, batch_size=batch_size)
        x, y = x.to(device), y.to(device)
        model.train()
        prediction = model(x)
        if prediction.shape != y.shape:
            raise RuntimeError(f"{name} prediction {tuple(prediction.shape)} != target {tuple(y.shape)}")
        loss = torch.nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _record_gradient_activity(model, connected_seen, nonzero_seen)
        optimizer.step()
        updates_by_city[city] += 1

    counts = parameter_summary(model)
    named_parameters = dict(model.named_parameters())
    counts["active_parameter_elements_connected_to_loss_at_least_once"] = int(
        sum(named_parameters[name].numel() for name, seen in connected_seen.items() if seen)
    )
    counts["active_parameter_tensors_connected_to_loss_at_least_once"] = int(sum(connected_seen.values()))
    counts["parameter_elements_with_nonzero_gradient_at_least_once"] = int(
        sum(named_parameters[name].numel() for name, seen in nonzero_seen.items() if seen)
    )
    counts["parameter_tensors_with_nonzero_gradient_at_least_once"] = int(sum(nonzero_seen.values()))
    return model, {
        "model": name,
        "seed": int(seed),
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "loss": "MSE on source-scaled volume only",
        "batch_size_windows": int(batch_size),
        "source_optimizer_updates_total": int(steps),
        "source_optimizer_updates_by_city": updates_by_city,
        "target_adapt_steps": 0,
        "parameters": counts,
    }


def predict_city(model: nn.Module, city: dict[str, Any], mean: np.ndarray,
                 std: np.ndarray, *, batch_size: int = 4,
                 device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = city["values"]
    starts = window_starts(TRAIN_END, VAL_END)
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            x = np.stack([values[s - CONTEXT:s] for s in batch_starts]).astype(np.float32)
            x = x.transpose(0, 1, 3, 2)
            x = (x - mean.reshape(1, 1, len(VARIABLES), 1)) / std.reshape(1, 1, len(VARIABLES), 1)
            tensor = torch.from_numpy(x).to(device)
            prediction = model(tensor).cpu().numpy()
            y = np.stack([values[s:s + HORIZON, :, 0] for s in batch_starts]).astype(np.float32)
            predictions.append(prediction)
            targets.append(y)
    pred_scaled = np.concatenate(predictions, axis=0)
    truth_raw = np.concatenate(targets, axis=0).astype(np.float32)
    pred_raw = (pred_scaled * std[0] + mean[0]).astype(np.float32)
    return pred_raw, truth_raw, starts


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error)))}


def code_provenance() -> dict[str, Any]:
    paths = [
        HERE,
        CHARGED / "city_adapter.py",
        EXTERNAL / "cross_city_adapter.py",
        REPRO / "baselines" / "phaseformer_adapter.py",
    ] + sorted((REPRO / "vendor" / "PhaseFormer" / "src").rglob("*.py"))
    files = {str(path.relative_to(ROOT)): _file_hash(path) for path in paths if path.is_file()}
    return {
        "code_sha256": files,
        "phaseformer_upstream_commit": PHASEFORMER_COMMIT,
        "runner_sha256": files[str(HERE.relative_to(ROOT))],
    }


def run(*, output_dir: Path, seeds: tuple[int, ...], steps: int,
        batch_size: int, learning_rate: float, device: str) -> dict[str, Any]:
    cities, data_prov = load_feature_arrays()
    all_results = []
    prediction_paths: list[str] = []
    state_paths: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in CITY_CODES:
        source_ids = tuple(city for city in CITY_CODES if city != target)
        mean, std = source_only_scalers(cities, source_ids)
        target_results = []
        for seed in seeds:
            fold_predictions: dict[str, np.ndarray] = {}
            fold_truth: np.ndarray | None = None
            starts: np.ndarray | None = None
            model_states = {}
            for name in MODELS:
                model, train_info = fit_model(
                    name, cities, source_ids, mean, std, seed=seed, steps=steps,
                    batch_size=batch_size, learning_rate=learning_rate, device=device,
                )
                prediction, truth, pred_starts = predict_city(
                    model, cities[target], mean, std, batch_size=batch_size, device=device,
                )
                if fold_truth is None:
                    fold_truth, starts = truth, pred_starts
                elif not np.array_equal(fold_truth, truth) or not np.array_equal(starts, pred_starts):
                    raise AssertionError("model prediction labels do not match")
                fold_predictions[name] = prediction
                row = {
                    **train_info,
                    "target_city": target,
                    "source_cities": list(source_ids),
                    "metrics_raw_volume": metrics(prediction, truth),
                    "validation_label_start_index": TRAIN_END,
                    "validation_label_end_exclusive": VAL_END,
                    "validation_windows": int(len(pred_starts)),
                    "site_count": int(len(cities[target]["node_ids"])),
                }
                target_results.append(row)
                model_states[name] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                del model
            prediction_file = output_dir / f"charged_covariate_{target.lower()}_seed{seed}_predictions.npz"
            np.savez_compressed(
                prediction_file,
                label_starts=starts,
                label_timestamps=np.asarray([
                    str(cities[target]["times"][int(start)]) for start in starts
                ]),
                site_ids=np.asarray(cities[target]["node_ids"], dtype=str),
                truth_volume_raw=fold_truth,
                prediction_hptn_raw=fold_predictions["hptn"],
                prediction_phaseformer_raw=fold_predictions["phaseformer"],
            )
            prediction_paths.append(str(prediction_file))
            state_file = output_dir / f"charged_covariate_{target.lower()}_seed{seed}_states.pt"
            torch.save(model_states, state_file)
            state_paths.append(str(state_file))
            all_results.extend(target_results)
    payload = {
        "experiment": "CHARGED v3 source-only LOCO, volume-duration HPTN vs matched-input PhaseFormer",
        "status": "completed",
        "protocol": {
            "cities": list(CITY_CODES), "context": CONTEXT, "horizon": HORIZON,
            "variables_order": list(VARIABLES), "target": "volume",
            "train_end_exclusive": TRAIN_END,
            "train_end_timestamp_exclusive": "2023-08-01 00:00:00",
            "validation_label_start": TRAIN_END,
            "validation_end_exclusive": VAL_END,
            "validation_end_timestamp_exclusive": "2023-09-01 00:00:00",
            "validation_month": "2023-08", "validation_windows_per_city": 721,
            "scaler": "one source-only global mean/std per variable; equal weight per source city; fitted on indices [0,2928)",
            "target_scaler_data_used": False, "target_adaptation_steps": 0,
            "source_optimizer_updates_per_model_target_seed": steps,
            "source_city_update_schedule": "round-robin source cities, one minibatch/update",
            "batch_size_windows": batch_size, "loss": "same MSE on standardized volume",
            "learning_rate": learning_rate, "seeds": list(seeds),
            "device": device, "remove_zero": False,
            "future_covariates_used": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "architecture": {
            "shared_phase_backbone": {
                "implementation": "pinned official PhaseFormer forward adapter",
                "sequence_length": CONTEXT, "prediction_length": HORIZON,
                "input_channels": len(VARIABLES), "latent_dim": 32,
                "phase_layers": 2, "attention_heads": 8,
                "dropout": 0.1, "routers": 1, "positional_embedding": True,
                "node_as_batch": True,
            },
            "phaseformer_control": "same two input variables and backbone; volume output channel only; official core has no cross-channel fusion",
            "hptn_cross_variable_attention_heads": 4,
            "hptn_additions": [
                "4-head attention across the two variable tokens at each node and forecast phase",
                "shared 2*32-to-32-to-1 transport head",
                "exact zero-mean node correction at each forecast horizon",
            ],
            "excluded_covariates_and_reason": {
                "source": "CHARGED Scientific Data paper, DOI 10.1038/s41597-025-05584-7",
                "e_price": "Excluded from the primary cross-city comparison because records are in city-local currency with no documented conversion, so magnitudes are not cross-city comparable; extreme values (for example SPO=800) also warrant data-quality sensitivity analysis.",
                "s_price": "Excluded from the primary cross-city comparison because records are in city-local currency with no documented conversion, so magnitudes are not cross-city comparable; consider a separately labeled within-city or currency-normalized sensitivity analysis.",
            },
        },
        "data_provenance": data_prov,
        "code_provenance": code_provenance(),
        "results": all_results,
        "prediction_files": prediction_paths,
        "state_files": state_paths,
    }
    summary = output_dir / "charged_covariate_hptn_phaseformer_loco_results.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["summary_json"] = str(summary)
    return payload


def cpu_smoke() -> dict[str, Any]:
    """Small synthetic forward/backward only; never reads CHARGED labels."""
    torch.manual_seed(0)
    x = torch.randn(2, CONTEXT, len(VARIABLES), 3, dtype=torch.float32)
    target = torch.randn(2, HORIZON, 3, dtype=torch.float32)
    summaries: dict[str, Any] = {}
    for name in MODELS:
        model = build_model(name).cpu().eval()
        prediction = model(x)
        if prediction.shape != target.shape:
            raise AssertionError(f"{name} shape {tuple(prediction.shape)} != {tuple(target.shape)}")
        if name == "hptn":
            node_x, batch, nodes = model._node_batch(x)
            base_all = model.backbone(node_x)
            base_volume = base_all[..., 0].reshape(batch, nodes, HORIZON).transpose(1, 2)
            node_mean = (prediction - base_volume).mean(dim=2)
            if not torch.allclose(node_mean, torch.zeros_like(node_mean), atol=1e-5):
                raise AssertionError("HPTN correction is not zero-mean over nodes")
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        nonzero = sum(
            int(parameter.numel()) for parameter in model.parameters()
            if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        )
        grad_tensors = [parameter for parameter in model.parameters() if parameter.grad is not None]
        summaries[name] = {
            **parameter_summary(model),
            "active_parameter_tensors_connected_to_loss": len(grad_tensors),
            "active_parameter_elements_connected_to_loss": int(sum(p.numel() for p in grad_tensors)),
            "nonzero_gradient_elements_in_smoke_batch": nonzero,
            "output_shape": list(prediction.shape),
            "cpu_backward": True,
        }
        del model
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="explicitly start all configured LOCO training runs")
    parser.add_argument("--smoke-only", action="store_true",
                        help="run synthetic CPU forward/backward and exit")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parent / "charged_covariate_hptn_phaseformer_loco")
    parser.add_argument("--seeds", default=",".join(map(str, SEED_DEFAULTS)))
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.smoke_only:
        print(json.dumps(cpu_smoke(), indent=2))
        return 0
    if not args.run:
        parser.print_help()
        print("\nNo data was loaded and no training was started. Pass --smoke-only for CPU shape checks or --run to execute.")
        return 0
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.steps != 512 or args.batch_size != 4:
        raise ValueError("protocol is frozen at 512 updates and batch size 4")
    result = run(output_dir=args.output_dir, seeds=seeds, steps=args.steps,
                 batch_size=args.batch_size, learning_rate=args.learning_rate,
                 device=args.device)
    print(result["summary_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

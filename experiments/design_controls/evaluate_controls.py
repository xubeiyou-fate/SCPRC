#!/usr/bin/env python3
"""Read-only, bounded-memory evaluation of the 72 supplementary control runs.

No runner, model, torch, or training module is imported. See ANALYSIS_RULES.md.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "energy_value" / "run_energy_evaluation.py"
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
VARIANTS = ("full_same_run", "phase_shared_strict", "stationwise_matched", "concat_project")
FULL = VARIANTS[0]
POSTPROCESSES = ("raw", "nonnegative_clip")
CORE = ("peak_magnitude_mae", "ramp_mae", "total_24h_mae", "peak_timing_mae_hours")
RNG_SEED = 20261117
# Numerator, denominator, square-root after pooling. Counts are per origin.
SPECS = {
    "station_rmse": ("station_sse", "station_count", True),
    "station_mae": ("station_sae", "station_count", False),
    "city_total_rmse": ("city_total_sse", "city_total_count", True),
    "city_total_mae": ("city_total_sae", "city_total_count", False),
    "peak_magnitude_mae": ("peak_magnitude_sae", "peak_count", False),
    "peak_magnitude_rmse": ("peak_magnitude_sse", "peak_count", True),
    "ramp_mae": ("ramp_sae", "ramp_count", False),
    "ramp_rmse": ("ramp_sse", "ramp_count", True),
    "total_24h_mae": ("total_24h_sae", "total_count", False),
    "total_24h_rmse": ("total_24h_sse", "total_count", True),
    "peak_timing_mae_hours": ("peak_timing_sae_hours", "peak_count", False),
}
PRIMITIVES = tuple(dict.fromkeys(x for spec in SPECS.values() for x in spec[:2]))
CSV_FIELDS = {
    "per_fold.csv": ["city", "seed", "variant", "postprocess", "n_origins", "horizon", "n_sites", *SPECS, *PRIMITIVES],
    "summaries.csv": ["city", "variant", "postprocess", "n_folds", "expected_folds", "n_cities", "complete", "estimator", *SPECS],
    "paired_per_fold.csv": ["city", "seed", "control", "reference", "postprocess", "metric", "control_error", "full_error",
                            "difference_control_minus_full", "fold_relative_gain_pct_for_full"],
    "paired_summary.csv": ["control", "reference", "postprocess", "metric", "n_folds", "n_cities", "complete", "estimator",
                           "control_mean", "full_mean", "difference_control_minus_full", "ratio_of_means_relative_gain_pct_for_full",
                           "mean_fold_relative_gain_pct_for_full", "undefined_relative_folds", "full_better_folds"],
    "uncertainty.csv": ["control", "reference", "postprocess", "metric", "difference_control_minus_full", "ci95_low", "ci95_high",
                        "interval_method", "bootstrap_replicates", "rng_seed", "n_cities", "n_seeds_per_city", "n_folds",
                        "block_length_origins", "interpretation"],
}


def fold_name(city, seed):
    return f"target_{city.lower()}_seed{seed}"


def finite(value, where="value"):
    """Reject nonfinite values recursively, including optional result metadata."""
    if isinstance(value, dict):
        for k, v in value.items():
            finite(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite(v, f"{where}[{i}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite {where}")
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"Nonfinite array {where}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    finite(value, str(path))
    return value


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    finite(value)
    with atomic_path(path) as tmp:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ["status"])
    finite(rows)
    with atomic_path(path) as tmp:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array, chunk=32):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    for start in range(0, len(array), chunk):
        h.update(np.ascontiguousarray(array[start:start + chunk]).tobytes())
    return h.hexdigest()


def load_reference(path=REFERENCE):
    """Compile only the two exact pure definitions; never execute module code."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    names = {"origin_scores", "metric_rows"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Reference pure metric definitions missing")
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["origin_scores"], scope["metric_rows"]


def chunk_scores(truth, prediction, origin_scores):
    """FP64 before subtraction, clipping, station sums, and all reductions."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[1] != 24:
        raise ValueError("Metrics require aligned [origins,24,sites] arrays")
    finite(truth, "truth chunk")
    finite(prediction, "prediction chunk")
    y = truth.sum(axis=2, dtype=np.float64)
    p = prediction.sum(axis=2, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = origin_scores(y, p)
        error = prediction - truth
        ramps = np.diff(p, axis=1) - np.diff(y, axis=1)
        city_error = p - y
        n, horizon, sites = truth.shape
        scores.update({
            "station_sse": np.sum(error * error, axis=(1, 2), dtype=np.float64),
            "station_sae": np.sum(np.abs(error), axis=(1, 2), dtype=np.float64),
            "station_count": np.full(n, horizon * sites, dtype=np.int64),
            "city_total_sse": np.sum(city_error * city_error, axis=1, dtype=np.float64),
            "city_total_sae": np.sum(np.abs(city_error), axis=1, dtype=np.float64),
            "city_total_count": np.full(n, horizon, dtype=np.int64),
            "ramp_sse": np.sum(ramps * ramps, axis=1, dtype=np.float64),
            "ramp_sae": np.sum(np.abs(ramps), axis=1, dtype=np.float64),
            "ramp_count": np.full(n, horizon - 1, dtype=np.int64),
            "peak_magnitude_sae": scores["peak_magnitude_mae"],
            "peak_magnitude_sse": scores["peak_signed_error"] ** 2,
            "peak_count": np.ones(n, dtype=np.int64),
            "peak_timing_sae_hours": scores["peak_timing_mae_hours"],
            "total_24h_sae": scores["total_24h_mae"],
            "total_24h_sse": scores["total_signed_error"] ** 2,
            "total_count": np.ones(n, dtype=np.int64),
            "negative_prediction_count": (prediction < 0).sum(axis=(1, 2), dtype=np.int64),
        })
        for metric, (num, den, root) in SPECS.items():
            value = scores[num] / scores[den]
            scores[metric] = np.sqrt(value) if root else value
    for name, values in scores.items():
        finite(values, name)
    return scores


def aggregate(primitives):
    totals = {k: np.sum(primitives[k], dtype=np.float64).item() for k in PRIMITIVES}
    result = {}
    for name, (num, den, root) in SPECS.items():
        value = totals[num] / totals[den]
        result[name] = math.sqrt(value) if root else value
    for k, value in totals.items():
        result[k] = int(value) if k.endswith("count") else value
    finite(result)
    return result


class FoldData:
    """NPZ truth is streamed to a temporary NPY, then mapped, never fully loaded."""
    def __init__(self, path, scratch, chunk):
        self.path = Path(path).resolve(strict=True)
        with zipfile.ZipFile(self.path) as archive:
            if "truth.npy" not in archive.namelist():
                raise ValueError(f"Missing native truth: {path}")
            fd, name = tempfile.mkstemp(prefix="controls-truth-", suffix=".npy", dir=scratch)
            os.close(fd)
            self.extracted = Path(name)
            with archive.open("truth.npy") as src, self.extracted.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=4 << 20)
        self.truth = np.load(self.extracted, mmap_mode="r", allow_pickle=False)
        if self.truth.dtype != np.dtype("float32") or self.truth.ndim != 3 or self.truth.shape[:2] != (721, 24) or self.truth.shape[2] < 1:
            raise ValueError(f"Expected native float32 truth [721,24,N]: {self.truth.shape}, {self.truth.dtype}")
        with np.load(self.path, allow_pickle=False) as z:
            self.starts = z["label_starts"]
            self.sites = z["site_ids"]
            self.times = {k: z[k] for k in ("origin_time", "target_time") if k in z}
        if self.starts.dtype.kind not in "iu" or not np.array_equal(self.starts, np.arange(2928, 3649)):
            raise ValueError("label_starts must be consecutive 2928..3648")
        if self.sites.ndim != 1 or len(self.sites) != self.truth.shape[2] or len(np.unique(self.sites)) != len(self.sites):
            raise ValueError("site_ids do not identify unique aligned stations")
        for key, times in self.times.items():
            shape = (721,) if key == "origin_time" else (721, 24)
            if times.shape != shape or times.dtype.kind != "M" or np.isnat(times).any():
                raise ValueError(f"Invalid optional {key} datetime64 array")
            if not np.all(np.diff(times, axis=0) == np.timedelta64(1, "h")):
                raise ValueError(f"Non-hourly {key} origins")
        if "target_time" in self.times and not np.all(np.diff(self.times["target_time"], axis=1) == np.timedelta64(1, "h")):
            raise ValueError("Target leads are not consecutive hours")
        if len(self.times) == 2 and not np.all(self.times["target_time"][:, 0] == self.times["origin_time"] + np.timedelta64(1, "h")):
            raise ValueError("Issue time / first target mismatch")
        for start in range(0, 721, chunk):
            finite(self.truth[start:start + chunk], "native truth")
        self.identity = {"truth": array_hash(self.truth, chunk), "label_starts": array_hash(self.starts),
                         "site_ids": array_hash(self.sites), **{k: array_hash(v) for k, v in self.times.items()}}
        self.sha256 = sha256_file(self.path)


def validate_result(result, city, seed, variant):
    if not isinstance(result, dict):
        raise ValueError("result.json must be an object")
    if result.get("status") != "completed":
        return False
    for key, expected in (("steps", 2048), ("window_exposures", 8192), ("optimizer_steps", 410), ("seed", seed)):
        if type(result.get(key)) is not int or result[key] != expected:
            raise ValueError(f"{key} must equal {expected}")
    if result.get("variant") != variant or str(result.get("target_city", "")).upper() != city:
        raise ValueError("Result city / variant does not match path")
    if type(result.get("parameter_count")) is not int or result["parameter_count"] <= 0:
        raise ValueError("Missing positive parameter_count")
    for key in ("initialization", "training", "metrics", "fold_protocol"):
        if not isinstance(result.get(key), dict):
            raise ValueError(f"Missing result object: {key}")
    initial_count = result["initialization"].get("parameter_count", result["parameter_count"])
    if initial_count != result["parameter_count"]:
        raise ValueError("Initialization parameter count mismatch")
    cfg = result["fold_protocol"]
    if "configured_source_minibatches" in result and result["configured_source_minibatches"] != 2048:
        raise ValueError("configured_source_minibatches must equal 2048")
    if "target_city" in cfg and str(cfg["target_city"]).upper() != city:
        raise ValueError("Fold protocol target_city mismatch")
    if "seed" in cfg and cfg["seed"] != seed:
        raise ValueError("Fold protocol seed mismatch")
    for key, expected in (("context", 168), ("horizon", 24), ("evaluation_origin_count", 721),
                          ("target_support_projection", False), ("target_release_gate", False)):
        if key in cfg and cfg[key] != expected:
            raise ValueError(f"Unexpected fold_protocol.{key}")
    return True


def optional_diagnostics(directory):
    values, warnings = {}, []
    for name in ("diagnostics.json", "run_config.json", "initialization.json", "training_progress.json"):
        path = directory / name
        if path.is_file():
            try:
                values[name] = read_json(path)
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f"{path}: {exc}")
    return values, warnings


def evaluate_run(data, pred_path, city, seed, variant, chunk, origin_scores, writer):
    prediction = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != data.truth.shape or prediction.dtype.kind != "f":
        raise ValueError(f"Invalid prediction shape / dtype: {prediction.shape}, {prediction.dtype}")
    primitives = {post: {k: [] for k in PRIMITIVES} for post in POSTPROCESSES}
    # Keep only 721 scalar sufficient statistics per model/mode, not trajectories.
    # The caller spools rows per run, so a bad late chunk cannot leak partial rows.
    for start in range(0, 721, chunk):
        stop = min(721, start + chunk)
        truth = np.asarray(data.truth[start:stop], dtype=np.float64)
        raw = np.asarray(prediction[start:stop], dtype=np.float64)
        for post in POSTPROCESSES:
            pred = raw if post == "raw" else np.maximum(raw, 0.0)
            scores = chunk_scores(truth, pred, origin_scores)
            for k in PRIMITIVES:
                primitives[post][k].append(scores[k])
            for i in range(stop - start):
                index = start + i
                row = {"city": city, "seed": seed, "variant": variant, "postprocess": post,
                       "origin_index": index, "label_start": int(data.starts[index]),
                       "origin_time": str(data.times["origin_time"][index]) if "origin_time" in data.times else "",
                       "target_start": str(data.times["target_time"][index, 0]) if "target_time" in data.times else "",
                       "target_end": str(data.times["target_time"][index, -1]) if "target_time" in data.times else ""}
                row.update({k: v[i].item() for k, v in scores.items()})
                if writer[0] is None:
                    writer[0] = csv.DictWriter(writer[1], fieldnames=list(row))
                    writer[0].writeheader()
                writer[0].writerow(row)
    rows = []
    for post in POSTPROCESSES:
        primitives[post] = {k: np.concatenate(v) for k, v in primitives[post].items()}
        rows.append({"city": city, "seed": seed, "variant": variant, "postprocess": post,
                     "n_origins": 721, "horizon": 24, "n_sites": int(data.truth.shape[2]),
                     **aggregate(primitives[post])})
    return rows, primitives


def relative_gain(control, full):
    return 100.0 * (control - full) / control if control != 0 else None


def make_summaries(rows):
    output = []
    for city in (*CITIES, "ALL"):
        for variant in VARIANTS:
            for post in POSTPROCESSES:
                group = [r for r in rows if r["variant"] == variant and r["postprocess"] == post
                         and (city == "ALL" or r["city"] == city)]
                if not group:
                    continue
                expected = 18 if city == "ALL" else 3
                complete = len(group) == expected
                output.append({"city": city, "variant": variant, "postprocess": post,
                               "n_folds": len(group), "expected_folds": expected,
                               "n_cities": len({r["city"] for r in group}), "complete": complete,
                               "estimator": "equal_18_fold_mean" if complete and city == "ALL" else
                               "equal_3_seed_mean" if complete else "equal_available_fold_mean_monitoring",
                               **{m: float(np.mean([r[m] for r in group], dtype=np.float64)) for m in SPECS}})
    return output


def make_pairs(rows):
    index = {(r["city"], r["seed"], r["variant"], r["postprocess"]): r for r in rows}
    folds, summary = [], []
    for control in VARIANTS[1:]:
        for post in POSTPROCESSES:
            for metric in SPECS:
                group = []
                for city in CITIES:
                    for seed in SEEDS:
                        a = index.get((city, seed, control, post))
                        b = index.get((city, seed, FULL, post))
                        if a is None or b is None:
                            continue
                        row = {"city": city, "seed": seed, "control": control, "reference": FULL,
                               "postprocess": post, "metric": metric, "control_error": a[metric],
                               "full_error": b[metric], "difference_control_minus_full": a[metric] - b[metric],
                               "fold_relative_gain_pct_for_full": relative_gain(a[metric], b[metric])}
                        group.append(row)
                        folds.append(row)
                if not group:
                    continue
                a_mean = float(np.mean([r["control_error"] for r in group], dtype=np.float64))
                b_mean = float(np.mean([r["full_error"] for r in group], dtype=np.float64))
                relative = [r["fold_relative_gain_pct_for_full"] for r in group]
                undefined = sum(v is None for v in relative)
                summary.append({"control": control, "reference": FULL, "postprocess": post,
                                "metric": metric, "n_folds": len(group),
                                "n_cities": len({r["city"] for r in group}), "complete": len(group) == 18,
                                "estimator": "equal_18_fold_mean" if len(group) == 18 else "equal_available_fold_mean_monitoring",
                                "control_mean": a_mean, "full_mean": b_mean,
                                "difference_control_minus_full": float(np.mean([r["difference_control_minus_full"] for r in group])),
                                "ratio_of_means_relative_gain_pct_for_full": relative_gain(a_mean, b_mean),
                                "mean_fold_relative_gain_pct_for_full": None if undefined else float(np.mean(relative)),
                                "undefined_relative_folds": undefined,
                                "full_better_folds": sum(r["difference_control_minus_full"] > 0 for r in group)})
    return folds, summary


def resampled_metrics(matrix, starts, block_length=168):
    """Sum noncircular blocks via prefix sums, then reconstruct every fold RMSE."""
    n = len(matrix)
    lengths = [min(block_length, n - j * block_length) for j in range(starts.shape[1])]
    if sum(lengths) != n or any(length <= 0 for length in lengths):
        raise ValueError("Block schedule must retain exactly n origins")
    if np.any(starts < 0) or np.any(starts > n - block_length):
        raise ValueError("Noncircular block start out of bounds")
    prefix = np.vstack([np.zeros((1, matrix.shape[1]), dtype=np.float64),
                        np.cumsum(matrix, axis=0, dtype=np.float64)])
    totals = np.zeros((len(starts), matrix.shape[1]), dtype=np.float64)
    for j, length in enumerate(lengths):
        totals += prefix[starts[:, j] + length] - prefix[starts[:, j]]
    columns = {k: i for i, k in enumerate(PRIMITIVES)}
    output = []
    for num, den, root in SPECS.values():
        value = totals[:, columns[num]] / totals[:, columns[den]]
        output.append(np.sqrt(value) if root else value)
    result = np.column_stack(output)
    finite(result, "resampled fold metrics")
    return result


def bootstrap(rows, caches, draws):
    """Intervals only for complete 18-fold pairs; draws shared across contrasts."""
    idx = {(r["city"], r["seed"], r["variant"], r["postprocess"]): r for r in rows}
    eligible = [v for v in VARIANTS[1:] if all((c, s, v, "raw") in idx and (c, s, FULL, "raw") in idx
                                            for c in CITIES for s in SEEDS)]
    if not eligible:
        return []
    metric_names = list(SPECS)
    rng = np.random.default_rng(RNG_SEED)
    city_draws = rng.integers(0, 6, size=(draws, 6))
    models = (FULL, *eligible)
    temporal = {(v, post): np.zeros((draws, len(SPECS)), dtype=np.float64)
                for v in models for post in POSTPROCESSES}
    # One city at a time; only scalar origin primitives, never site trajectories.
    for city_index, city in enumerate(CITIES):
        time_rng = np.random.default_rng(np.random.SeedSequence([RNG_SEED, 1, city_index]))
        starts = time_rng.integers(0, 721 - 168 + 1, size=(draws, math.ceil(721 / 168)))
        for seed in SEEDS:
            for variant in models:
                with np.load(caches[(city, seed, variant)], allow_pickle=False) as z:
                    for post in POSTPROCESSES:
                        matrix = np.column_stack([z[f"{post}__{k}"] for k in PRIMITIVES])
                        temporal[(variant, post)] += resampled_metrics(matrix, starts) / 18.0
        print(f"BOOTSTRAP temporal {city} ({draws} shared draws)", flush=True)
    result = []
    for variant in eligible:
        for post in POSTPROCESSES:
            delta = np.asarray([[idx[(c, s, variant, post)][m] - idx[(c, s, FULL, post)][m]
                                 for m in SPECS] for c in CITIES for s in SEEDS], dtype=np.float64)
            city_delta = delta.reshape(6, 3, -1).mean(axis=1)
            distributions = {
                "six_city_cluster_bootstrap_all_three_seeds": city_delta[city_draws].mean(axis=1),
                "paired_168_origin_noncircular_moving_block_fixed_cities_seeds": temporal[(variant, post)] - temporal[(FULL, post)],
            }
            for method, samples in distributions.items():
                finite(samples, method)
                bounds = np.quantile(samples, [.025, .975], axis=0)
                for j, metric in enumerate(metric_names):
                    result.append({"control": variant, "reference": FULL, "postprocess": post,
                                   "metric": metric, "difference_control_minus_full": float(delta[:, j].mean()),
                                   "ci95_low": float(bounds[0, j]), "ci95_high": float(bounds[1, j]),
                                   "interval_method": method, "bootstrap_replicates": draws, "rng_seed": RNG_SEED,
                                   "n_cities": 6, "n_seeds_per_city": 3, "n_folds": 18,
                                   "block_length_origins": 168 if method.startswith("paired_168") else None,
                                   "interpretation": "exploratory pointwise conditional sensitivity; no causal or familywise claim"})
    return result


def require_matplotlib_panel_alignment(fig, axes, json_out, tolerance_pt=1.5):
    """Portable rendered-geometry gate for this script's explicit 2-by-2 grids."""
    fig.canvas.draw()
    width, height = fig.get_size_inches() * 72.0
    bounds = np.asarray([[a.get_position().x0 * width, a.get_position().y0 * height,
                          a.get_position().x1 * width, a.get_position().y1 * height] for a in axes])
    deviations = []
    for left, right in ((0, 1), (2, 3)):
        deviations.extend(abs(bounds[left, [1, 3]] - bounds[right, [1, 3]]))
        deviations.append(abs((bounds[left, 2] - bounds[left, 0]) - (bounds[right, 2] - bounds[right, 0])))
    for top, bottom in ((0, 2), (1, 3)):
        deviations.extend(abs(bounds[top, [0, 2]] - bounds[bottom, [0, 2]]))
        deviations.append(abs((bounds[top, 3] - bounds[top, 1]) - (bounds[bottom, 3] - bounds[bottom, 1])))
    passed = max(deviations, default=0) <= tolerance_pt
    report = {"status": "PASS" if passed else "FIX BEFORE DELIVERY", "tolerance_pt": tolerance_pt,
              "max_deviation_pt": float(max(deviations, default=0)), "plot_area_bounds_pt": bounds.tolist(),
              "scope": "final rendered equal 2x2 axes; no claim of text collision audit"}
    write_json(json_out, report)
    if not passed:
        raise ValueError(f"Rendered panel alignment failed: {report}")


def make_plots(out, paired, summaries, intervals, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not paired:
        return []
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False})
    labels = {"phase_shared_strict": "Phase shared", "stationwise_matched": "Stationwise", "concat_project": "Concat-project"}
    titles = {"peak_magnitude_mae": "Peak magnitude MAE", "ramp_mae": "Ramp MAE",
              "total_24h_mae": "24 h total MAE", "peak_timing_mae_hours": "Peak timing MAE (h)"}
    colors = ("#3D6B99", "#A56842", "#776393")
    table = {(r["control"], r["postprocess"], r["metric"]): r for r in summaries}
    ci = {(r["control"], r["postprocess"], r["metric"]): r for r in intervals
          if r["interval_method"].startswith("six_city")}
    suffix = "" if complete else " — MONITORING"
    files = []

    def save(fig, axes, stem):
        require_matplotlib_panel_alignment(fig, axes, out / f"{stem}.alignment.json")
        for suffix in (".pdf", ".svg", ".png"):
            path = out / f"{stem}{suffix}"
            with atomic_path(path) as tmp:
                fig.savefig(tmp, format=suffix[1:], dpi=300, facecolor="white")
            files.append(str(path))
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=.16, right=.97, bottom=.15, top=.86, wspace=.75, hspace=.65)
    for panel, (ax, metric) in enumerate(zip(axs.flat, CORE)):
        ax.axvline(0, color="0.7", lw=.7, zorder=0)
        for j, (variant, color) in enumerate(zip(VARIANTS[1:], colors)):
            row = table.get((variant, "raw", metric))
            if row is None:
                continue
            for city_index, city in enumerate(CITIES):
                ds = [r["difference_control_minus_full"] for r in paired if r["city"] == city and
                      r["control"] == variant and r["postprocess"] == "raw" and r["metric"] == metric]
                if ds:
                    ax.plot(np.mean(ds), j + (city_index - 2.5) * .045, "o", ms=3, color=color, alpha=.45)
            interval = ci.get((variant, "raw", metric))
            if interval:
                ax.plot([interval["ci95_low"], interval["ci95_high"]], [j, j], color=color, lw=1.5)
            ax.plot(row["difference_control_minus_full"], j, "D", ms=4, color=color)
        ax.set(yticks=range(3), yticklabels=[labels[v] for v in VARIANTS[1:]], ylim=(2.5, -.5),
               title=titles[metric], xlabel="Control − Full error")
        ax.text(-.24, 1.11, chr(97 + panel), transform=ax.transAxes, weight="bold", fontsize=10)
    fig.suptitle("Raw paired energy errors" + suffix, y=.97, fontsize=11)
    fig.text(.5, .92, "Positive differences favor Full; raw predictions are primary", ha="center", fontsize=8)
    fig.text(.5, .045, "Dots: available city seed-means; diamonds: equal-fold means.\n"
             "Lines: exploratory 95% city-cluster intervals only for complete 6-city × 3-seed pairs.",
             ha="center", fontsize=7)
    save(fig, list(axs.flat), "main_paired_raw_energy")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    fig.subplots_adjust(left=.12, right=.97, bottom=.20, top=.85, wspace=.47, hspace=.70)
    for panel, (ax, metric) in enumerate(zip(axs.flat, CORE)):
        ax.axhline(0, color="0.75", lw=.7, zorder=0)
        ax.axvline(0, color="0.75", lw=.7, zorder=0)
        for variant, color in zip(VARIANTS[1:], colors):
            coords = []
            for post, marker in zip(POSTPROCESSES, ("o", "^")):
                x = table.get((variant, post, "station_rmse"))
                y = table.get((variant, post, metric))
                if x is None or y is None:
                    continue
                xv, yv = x["difference_control_minus_full"], y["difference_control_minus_full"]
                coords.append((xv, yv))
                xci, yci = ci.get((variant, post, "station_rmse")), ci.get((variant, post, metric))
                if xci and yci:
                    ax.plot([xci["ci95_low"], xci["ci95_high"]], [yv, yv], color=color, lw=.65, alpha=.6)
                    ax.plot([xv, xv], [yci["ci95_low"], yci["ci95_high"]], color=color, lw=.65, alpha=.6)
                ax.plot(xv, yv, marker, ms=4, color=color)
            if len(coords) == 2:
                ax.plot(*np.asarray(coords).T, color=color, lw=.7, alpha=.7)
        ax.set(title=titles[metric], xlabel="Station RMSE: control − Full", ylabel="Energy error: control − Full")
        ax.text(-.16, 1.12, chr(97 + panel), transform=ax.transAxes, weight="bold", fontsize=10)
    handles = [Line2D([], [], marker="s", color=c, ls="none", label=labels[v])
               for v, c in zip(VARIANTS[1:], colors)]
    handles += [Line2D([], [], marker=m, color="0.2", ls="none", label=label)
                for m, label in (("o", "Raw"), ("^", "Nonnegative clip"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .055), fontsize=7)
    fig.suptitle("Station / energy tradeoff: raw and clipped" + suffix, y=.97, fontsize=11)
    fig.text(.5, .92, "Equal-fold paired means; segments connect postprocessing modes", ha="center", fontsize=8)
    fig.text(.5, .015, "Bars: exploratory 95% city-cluster intervals for complete pairs; source: paired summary / uncertainty tables.",
             ha="center", fontsize=7)
    save(fig, list(axs.flat), "tradeoff_raw_clip")
    return files


def evaluate(args):
    evaluation_started = time.perf_counter()
    root = args.root.resolve()
    runs = (args.runs if args.runs is not None else root / "runs").resolve()
    out = (args.out if args.out is not None else root / "analysis").resolve()
    data_root = (args.fold_data_root if args.fold_data_root is not None else runs).resolve()
    if out == runs or runs in out.parents or out == root or out == REFERENCE.parent:
        raise ValueError("Analysis output must be separate from runs, experiment root, and prior energy outputs")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {"schema_version": 1, "status": "running", "evaluation_completed": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "runs": str(runs),
                "out": str(out), "fold_data_root": str(data_root), "expected_runs": 72,
                "expected_folds_per_variant": 18, "cities": list(CITIES), "seeds": list(SEEDS),
                "variants": list(VARIANTS), "postprocesses": list(POSTPROCESSES), "primary_postprocess": "raw",
                "csv_schema": CSV_FIELDS,
                "allow_partial": args.allow_partial, "chunk_size_origins": args.chunk_size,
                "bootstrap_replicates": args.bootstrap_draws, "bootstrap_rng_seed": RNG_SEED,
                "nondefault_draw_count": args.bootstrap_draws != 10000, "block_length_origins": 168,
                "primary_estimator": "equal mean of 18 fold metrics",
                "paired_difference": "control minus Full; positive favors Full",
                "interval_interpretation": "exploratory pointwise conditional sensitivity; no causal or familywise claims",
                "p_values_computed": False, "training_invoked": False, "input_files_modified": False,
                "precision": "native float32 truth; FP64 arithmetic and reductions",
                "reference": {"path": str(REFERENCE), "sha256": sha256_file(REFERENCE),
                              "pure_functions_only": ["origin_scores", "metric_rows"]},
                "evaluator_sha256": sha256_file(__file__), "software": {"python": platform.python_version(), "numpy": np.__version__},
                "runs_snapshot": [], "warnings": [], "outputs": []}
    write_json(manifest_path, manifest)
    try:
        origin_scores, _ = load_reference()
        rows, caches = [], {}
        data_cache, city_identity, fold_protocols, backbone_hashes = {}, {}, {}, {}
        eligible_count = 0
        with tempfile.TemporaryDirectory(prefix="supp-controls-eval-") as scratch:
            with atomic_path(out / "per_origin.csv.gz") as origin_tmp:
                with gzip.open(origin_tmp, "wt", newline="", encoding="utf-8") as origin_output:
                    header_written = False
                    for city in CITIES:
                        for seed in SEEDS:
                            for variant in VARIANTS:
                                directory = runs / fold_name(city, seed) / variant
                                result_path = directory / "result.json"
                                record = {"city": city, "seed": seed, "variant": variant, "result_path": str(result_path)}
                                manifest["runs_snapshot"].append(record)
                                if not result_path.is_file():
                                    record.update(status="missing", reason="result.json not yet available")
                                    continue
                                try:
                                    result = read_json(result_path)
                                    if not isinstance(result, dict):
                                        raise ValueError("result.json must contain an object")
                                    record["runner_status"] = result.get("status")
                                    # Smoke is ignored before budget and other metadata checks.
                                    if result.get("status") != "completed":
                                        record.update(status="ignored_smoke" if result.get("status") == "completed_smoke" else "not_ready",
                                                      reason="Only full-budget status completed is eligible")
                                        continue
                                    validate_result(result, city, seed, variant)
                                    eligible_count += 1
                                    pred = directory / "prediction_raw.npy"
                                    if not pred.is_file():
                                        candidates = sorted(directory.rglob("prediction_raw.npy"))
                                        if len(candidates) != 1:
                                            raise ValueError(f"Expected exactly one prediction_raw.npy; found {len(candidates)}")
                                        pred = candidates[0]
                                    data_path = data_root / fold_name(city, seed) / "fold_data.npz"
                                    resolved = data_path.resolve(strict=True)
                                    if resolved not in data_cache:
                                        data_cache[resolved] = FoldData(data_path, scratch, args.chunk_size)
                                    data = data_cache[resolved]
                                    if city in city_identity and city_identity[city] != data.identity:
                                        raise ValueError("Truth / station / origin identity differs across seeds in this city")
                                    key = (city, seed)
                                    cfg = result["fold_protocol"]
                                    if key in fold_protocols and fold_protocols[key] != cfg:
                                        raise ValueError("Fold protocol differs across variants")
                                    expected_data_hash = cfg.get("fold_data_sha256")
                                    if expected_data_hash and expected_data_hash != data.sha256:
                                        raise ValueError("Fold-data checksum mismatch")
                                    backbone = result["initialization"].get("backbone_state_sha256")
                                    if backbone and key in backbone_hashes and backbone_hashes[key] != backbone:
                                        raise ValueError("Backbone initialization differs across variants")
                                    before = (pred.stat().st_size, pred.stat().st_mtime_ns)
                                    prediction_hash = sha256_file(pred)
                                    if result.get("predictions_sha256") and result["predictions_sha256"] != prediction_hash:
                                        raise ValueError("Prediction checksum mismatch")
                                    result_hash = sha256_file(result_path)
                                    spool = Path(scratch) / "origin_rows.csv"
                                    with spool.open("w", newline="", encoding="utf-8") as stream:
                                        these_rows, primitives = evaluate_run(data, pred, city, seed, variant,
                                                                             args.chunk_size, origin_scores, [None, stream])
                                    if before != (pred.stat().st_size, pred.stat().st_mtime_ns) or result_hash != sha256_file(result_path):
                                        raise ValueError("Run changed while being evaluated; retry snapshot")
                                    cache = out / "origin_primitives" / f"{fold_name(city, seed)}__{variant}.npz"
                                    with atomic_path(cache) as tmp:
                                        with tmp.open("wb") as stream:
                                            np.savez_compressed(stream, label_starts=data.starts,
                                                                **{f"{post}__{k}": v for post, content in primitives.items() for k, v in content.items()})
                                    with spool.open(encoding="utf-8") as stream:
                                        header = stream.readline()
                                        if not header_written:
                                            origin_output.write(header)
                                            header_written = True
                                        shutil.copyfileobj(stream, origin_output, length=1 << 20)
                                    rows.extend(these_rows)
                                    caches[(city, seed, variant)] = cache
                                    city_identity[city] = data.identity
                                    fold_protocols[key] = cfg
                                    if backbone:
                                        backbone_hashes[key] = backbone
                                    diagnostics, warnings = optional_diagnostics(directory)
                                    manifest["warnings"].extend(warnings)
                                    record.update(status="evaluated", prediction_path=str(pred), prediction_sha256=prediction_hash,
                                                  prediction_shape=list(data.truth.shape), result_sha256=result_hash,
                                                  fold_data_path=str(data_path), resolved_fold_data_path=str(resolved),
                                                  fold_data_sha256=data.sha256, alignment_hashes=data.identity,
                                                  metadata=result, optional_diagnostics=diagnostics,
                                                  primitive_cache=str(cache), primitive_cache_sha256=sha256_file(cache))
                                    print(f"EVALUATED {city} seed={seed} {variant} raw+clip", flush=True)
                                except (OSError, ValueError, KeyError, TypeError, EOFError, FloatingPointError, zipfile.BadZipFile) as exc:
                                    record.update(status="invalid", reason=f"{type(exc).__name__}: {exc}")
                                    print(f"NOT READY {city} seed={seed} {variant}: {exc}", file=sys.stderr, flush=True)
                    if not header_written:
                        dummy = np.zeros((1, 24, 1), dtype=np.float64)
                        fields = ["city", "seed", "variant", "postprocess", "origin_index", "label_start", "origin_time", "target_start", "target_end",
                                  *chunk_scores(dummy, dummy, origin_scores)]
                        csv.DictWriter(origin_output, fieldnames=fields).writeheader()
        summaries = make_summaries(rows)
        paired, pair_summary = make_pairs(rows)
        complete = len(caches) == 72
        intervals = bootstrap(rows, caches, args.bootstrap_draws)
        for name, table in (("per_fold.csv", rows), ("summaries.csv", summaries),
                            ("paired_per_fold.csv", paired), ("paired_summary.csv", pair_summary), ("uncertainty.csv", intervals)):
            write_csv(out / name, table, CSV_FIELDS[name])
        plots = [] if args.no_plots else make_plots(out, paired, pair_summary, intervals, complete)
        manifest.update(status="complete" if complete else "partial", evaluation_completed=complete,
                        finished_at_utc=datetime.now(timezone.utc).isoformat(), eligible_completed_runs=eligible_count,
                        elapsed_seconds=round(time.perf_counter() - evaluation_started, 6),
                        evaluated_runs=len(caches), fold_metric_rows=len(rows), origin_metric_rows=len(caches) * 721 * 2,
                        invalid_runs=sum(r["status"] == "invalid" for r in manifest["runs_snapshot"]),
                        ignored_smoke_runs=sum(r["status"] == "ignored_smoke" for r in manifest["runs_snapshot"]),
                        pending_runs=sum(r["status"] in ("missing", "not_ready") for r in manifest["runs_snapshot"]),
                        all_numeric_outputs_finite=True, interval_rows=len(intervals),
                        complete_paired_comparisons=[v for v in VARIANTS[1:] if any(r["control"] == v and r["complete"] for r in pair_summary)],
                        plots_enabled=not args.no_plots,
                        outputs=[str(out / n) for n in ("per_fold.csv", "per_origin.csv.gz", "summaries.csv", "paired_per_fold.csv",
                                                         "paired_summary.csv", "uncertainty.csv")] + plots)
        manifest["output_sha256"] = {p: sha256_file(p) for p in manifest["outputs"]}
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "evaluated_runs": len(caches), "expected_runs": 72,
                          "manifest": str(manifest_path)}, allow_nan=False), flush=True)
        return 0 if complete or args.allow_partial else 2
    except Exception as exc:
        manifest.update(status="failed", evaluation_completed=False, failure=f"{type(exc).__name__}: {exc}")
        write_json(manifest_path, manifest)
        raise


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=HERE, help="Experiment root; defaults to this script's directory")
    ap.add_argument("--runs", type=Path, help="Runs directory containing target_<city>_seed<seed> subdirectories")
    ap.add_argument("--fold-data-root", type=Path, help="Alternate root of target_<city>_seed<seed>/fold_data.npz")
    ap.add_argument("--out", type=Path, help="Analysis output directory; default <root>/analysis")
    ap.add_argument("--chunk-size", type=int, default=16, help="Origins per FP64 chunk (default: 16)")
    ap.add_argument("--bootstrap-draws", type=int, default=10000, help="Replicates per sensitivity scheme (default: 10000)")
    ap.add_argument("--allow-partial", action="store_true", help="Monitor available full-budget runs without claiming completeness")
    ap.add_argument("--no-plots", action="store_true", help="Export numerical analysis without rendering figures")
    ap.add_argument("--self-test", action="store_true", help="Verify using synthetic temporary files; no training")
    return ap


def self_test():
    """Synthetic regression / integration checks; all artifacts live under /tmp."""
    started = time.perf_counter()
    assert parser().parse_args([]).root == HERE
    origin_scores, metric_rows = load_reference()
    rng = np.random.default_rng(1234)
    truth = rng.uniform(-1, 3, size=(721, 24, 2)).astype(np.float32)
    prediction = (truth + rng.normal(0, .4, truth.shape)).astype(np.float32)
    checks = []
    for post in POSTPROCESSES:
        pred = prediction if post == "raw" else np.maximum(prediction, 0)
        chunks = [chunk_scores(truth[i:i + 19], pred[i:i + 19], origin_scores) for i in range(0, 721, 19)]
        primitives = {k: np.concatenate([c[k] for c in chunks]) for k in PRIMITIVES}
        result = aggregate(primitives)
        reference, _ = metric_rows(truth, pred, "SYNTHETIC", 13, "test", "synthetic_only")
        for metric in SPECS:
            if metric in reference:
                np.testing.assert_allclose(result[metric], reference[metric], rtol=2e-13, atol=2e-13)
        error = pred.astype(np.float64) - truth.astype(np.float64)
        np.testing.assert_allclose(result["station_rmse"], np.sqrt(np.mean(error ** 2)), rtol=2e-13)
        np.testing.assert_allclose(result["station_mae"], np.mean(np.abs(error)), rtol=2e-13)
        assert result["station_count"] == 721 * 24 * 2 and result["ramp_count"] == 721 * 23
    checks.append("raw/clipped chunked FP64 equivalence to both exact reference functions")

    zero = np.zeros((2, 24, 1), dtype=np.float32)
    slope = np.broadcast_to(np.arange(24, dtype=np.float32)[None, :, None], zero.shape)
    scores = chunk_scores(zero, slope, origin_scores)
    assert np.array_equal(scores["peak_timing_mae_hours"], [23, 23])
    assert np.array_equal(scores["ramp_mae"], [1, 1]) and np.array_equal(scores["ramp_rmse"], [1, 1])
    tie = np.ones_like(zero)
    assert np.array_equal(chunk_scores(zero, tie, origin_scores)["predicted_peak_hour"], [0, 0])
    unequal = np.ones_like(zero)
    unequal[1] *= 3
    assert math.isclose(aggregate(chunk_scores(zero, unequal, origin_scores))["station_rmse"], math.sqrt(5))
    assert relative_gain(0, 0) is None
    checks.append("first-argmax ties, noncircular timing, 23 ramps, pooled rather than mean RMSE, zero denominators")

    primitive = chunk_scores(truth, prediction, origin_scores)
    matrix = np.column_stack([primitive[k] for k in PRIMITIVES])
    starts = np.asarray([[0, 553, 10, 300, 100], [553, 0, 50, 500, 200]])
    actual = resampled_metrics(matrix, starts)
    for i, draw in enumerate(starts):
        indices = np.concatenate([np.arange(s, s + 168) for s in draw])[:721]
        expected = aggregate({k: primitive[k][indices] for k in PRIMITIVES})
        np.testing.assert_allclose(actual[i], [expected[m] for m in SPECS], rtol=1e-12, atol=1e-12)
    checks.append("168-origin noncircular prefix sums match explicit paired-index pooling, including RMSE")
    for bad in (float("nan"), float("inf")):
        try:
            finite({"test": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite JSON accepted")
    bad_prediction = prediction.copy()
    bad_prediction[-1, -1, -1] = np.nan
    try:
        chunk_scores(truth, bad_prediction, origin_scores)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite prediction accepted")
    checks.append("finite JSON and late-chunk nonfinite input rejection")

    with tempfile.TemporaryDirectory(prefix="controls-selftest-", dir="/tmp") as directory:
        root = Path(directory)
        args = parser().parse_args(["--root", str(root), "--allow-partial", "--no-plots", "--bootstrap-draws", "31"])
        assert evaluate(args) == 0
        manifest = read_json(root / "analysis/manifest.json")
        assert not manifest["evaluation_completed"] and manifest["evaluated_runs"] == 0
        args.allow_partial = False
        assert evaluate(args) == 2
        # Explicitly selecting smoke still does not admit completed_smoke.
        smoke = root / "smoke/runs" / fold_name("AMS", 13) / FULL
        smoke.mkdir(parents=True)
        write_json(smoke / "result.json", {"status": "completed_smoke"})
        args.allow_partial = True
        args.runs = root / "smoke/runs"
        assert evaluate(args) == 0
        manifest = read_json(root / "analysis/manifest.json")
        assert manifest["ignored_smoke_runs"] == 1 and manifest["evaluated_runs"] == 0
        args.runs = None
        checks.append("empty partial / strict readiness; explicit smoke-root exclusion")

        # Small site count, full contract dimensions: 72 synthetic files, no models.
        times = np.datetime64("2023-07-31T23", "h") + np.arange(721).astype("timedelta64[h]")
        for city_index, city in enumerate(CITIES):
            data = root / "data" / f"{city}.npz"
            data.parent.mkdir(exist_ok=True)
            t = rng.uniform(-.2, 2 + city_index * .1, (721, 24, 1)).astype(np.float32)
            np.savez(data, truth=t, truth_raw=t, label_starts=np.arange(2928, 3649), site_ids=np.asarray([f"{city}_1"]),
                     origin_time=times, target_time=times[:, None] + np.arange(1, 25).astype("timedelta64[h]")[None, :])
            for seed in SEEDS:
                fold = root / "runs" / fold_name(city, seed)
                fold.mkdir(parents=True)
                (fold / "fold_data.npz").symlink_to(data)
                cfg = {"context": 168, "horizon": 24, "evaluation_origin_count": 721, "seed": seed,
                       "target_city": city, "fold_data_sha256": sha256_file(data)}
                for variant_index, variant in enumerate(VARIANTS):
                    run = fold / variant
                    run.mkdir()
                    p = (t + (variant_index + 1) * .05 * np.sin(np.arange(24))[None, :, None]).astype(np.float32)
                    np.save(run / "prediction_raw.npy", p)
                    write_json(run / "result.json", {"status": "completed", "steps": 2048,
                               "window_exposures": 8192, "optimizer_steps": 410, "variant": variant,
                               "seed": seed, "target_city": city, "parameter_count": 100 + variant_index,
                               "initialization": {"parameter_count": 100 + variant_index, "backbone_state_sha256": f"test-{seed}"},
                               "training": {"synthetic_verification_only": True}, "metrics": {}, "fold_protocol": cfg,
                               "predictions_sha256": sha256_file(run / "prediction_raw.npy")})
        # Best-effort diagnostic failure must not change completeness of valid runs.
        first = root / "runs" / fold_name("AMS", 13) / FULL
        (first / "diagnostics.json").write_text('{"nonfinite": NaN}', encoding="utf-8")
        args.allow_partial = False
        args.no_plots = False
        args.chunk_size = 37
        assert evaluate(args) == 0
        manifest = read_json(root / "analysis/manifest.json")
        assert manifest["evaluation_completed"] and manifest["evaluated_runs"] == 72
        assert manifest["fold_metric_rows"] == 144 and manifest["origin_metric_rows"] == 103824
        assert manifest["warnings"] and manifest["interval_rows"] == 132
        with (root / "analysis/summaries.csv").open() as stream:
            summary_rows = list(csv.DictReader(stream))
        assert all(r["n_folds"] == "18" for r in summary_rows if r["city"] == "ALL")
        for stem in ("main_paired_raw_energy", "tradeoff_raw_clip"):
            assert read_json(root / "analysis" / f"{stem}.alignment.json")["status"] == "PASS"
            for ext in ("pdf", "svg", "png"):
                assert (root / "analysis" / f"{stem}.{ext}").stat().st_size > 1000
        with gzip.open(root / "analysis/per_origin.csv.gz", "rt") as stream:
            assert sum(1 for _ in stream) == 103825
        checks.append("full 72-run schema / symlink / datetime / paired exports / plots / optional diagnostic failure integration")

        with (root / "analysis/per_fold.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["seed"] = int(row["seed"])
            for metric in SPECS:
                row[metric] = float(row[metric])
        caches = {(r["city"], r["seed"], r["variant"]): Path(r["primitive_cache"]) for r in manifest["runs_snapshot"]}
        a = bootstrap(rows, caches, 31)
        b = bootstrap(rows, caches, 31)
        assert a == b
        # Equal seed predictions give identical paired curves; shared draws preserve that.
        assert all(math.isfinite(r["ci95_low"]) and r["ci95_low"] <= r["ci95_high"] for r in a)
        benchmark_start = time.perf_counter()
        bootstrap(rows, caches, 10000)
        benchmark_seconds = time.perf_counter() - benchmark_start
        checks.append(f"deterministic shared city / temporal draws; 10000-draw synthetic bootstrap {benchmark_seconds:.2f}s")
        result_path = first / "result.json"
        bad = read_json(result_path)
        bad["optimizer_steps"] = 409
        try:
            validate_result(bad, "AMS", 13, FULL)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid budget accepted")
        checks.append("incorrect completed-run optimizer budget rejected")
        # Optionally retain only synthetic figure previews outside the repository for QA.
        preview = os.environ.get("CONTROLS_SELFTEST_PREVIEW_DIR")
        if preview:
            destination = Path(preview).resolve()
            if HERE.parents[1] == destination or HERE.parents[1] in destination.parents:
                raise ValueError("Synthetic previews must be outside the repository")
            destination.mkdir(parents=True, exist_ok=True)
            for p in (root / "analysis").glob("*.pdf"):
                shutil.copyfile(p, destination / p.name)
            for p in (root / "analysis").glob("*.png"):
                shutil.copyfile(p, destination / p.name)
    report = {"status": "passed", "synthetic_only": True, "training_invoked": False,
              "checks": checks, "seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.chunk_size < 1 or args.bootstrap_draws < 2:
        parser().error("--chunk-size must be positive and --bootstrap-draws at least 2")
    return self_test() if args.self_test else evaluate(args)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())

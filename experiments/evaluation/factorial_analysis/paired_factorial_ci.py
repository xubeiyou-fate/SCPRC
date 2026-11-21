#!/usr/bin/env python3
"""Paired Exact/Full and 2x2 PCGrad x relation analysis.

The four arms are
  Original          = relation off, PCGrad off
  Exact-PCGrad      = relation off, PCGrad on
  Full-no-PCGrad    = relation on, PCGrad off
  Full              = relation on, PCGrad on

This script is deliberately prediction based.  It checks truth/origin/site
alignment before scoring and refuses to create a factorial result until all
18 Full-no-PCGrad folds exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
OUT = Path(__file__).resolve().parent
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)
REPLICATES = 10000
RNG_SEED = 20261031
ORIGINAL_ROOT = ROOT / "experiments/forecasting/p2_no_pcgrad_phaseformer"
EXACT_ROOT = ROOT / 'experiments/forecasting/h24_framework'
FULL_NOPCGRAD_ROOT = ROOT / "experiments/evaluation/scprc_mean_gradient"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    err = pred.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(err * err))), float(np.mean(np.abs(err)))


def load_arm(path: Path, preferred: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        for key in preferred:
            if key in z.files:
                return z[key], {name: z[name] for name in ("label_starts", "truth_raw", "site_ids") if name in z.files}
    raise KeyError(f"none of {preferred} found in {path}; keys={z.files}")


def bootstrap(rows: list[dict], left: str, right: str, metric: str, rng: np.random.Generator) -> dict:
    a = np.asarray([row[f"{left}_{metric}"] for row in rows], dtype=np.float64)
    b = np.asarray([row[f"{right}_{metric}"] for row in rows], dtype=np.float64)
    n = len(rows)
    ids = rng.integers(0, n, size=(REPLICATES, n))
    delta = b[ids].mean(axis=1) - a[ids].mean(axis=1)
    relative = 100.0 * (b[ids].mean(axis=1) / np.maximum(a[ids].mean(axis=1), 1e-12) - 1.0)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "left_point": float(a.mean()),
        "right_point": float(b.mean()),
        "difference_right_minus_left": float(b.mean() - a.mean()),
        "difference_ci95": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))],
        "relative_change_percent": float(100.0 * (b.mean() / max(a.mean(), 1e-12) - 1.0)),
        "relative_change_ci95_percent": [float(np.quantile(relative, .025)), float(np.quantile(relative, .975))],
        "right_better_folds": int(np.sum(b < a)),
        "folds": n,
        "bootstrap_replicates": REPLICATES,
        "unit": "paired city x seed fold",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    hashes: dict[str, str] = {}
    for city in CITIES:
        for seed in SEEDS:
            key = f"target_{city.lower()}_seed{seed}"
            paths = {
                "original": ORIGINAL_ROOT / key / "predictions.npz",
                "exact": EXACT_ROOT / key / "predictions.npz",
                # Use the canonical Full anchor for the formal comparison and
                # factorial interaction.  The independent-switch Full row in
                # p2_components remains a separate ablation run identity.
                "full": EXACT_ROOT / key / "predictions.npz",
                "full_no_pcgrad": FULL_NOPCGRAD_ROOT / key / "predictions.npz",
            }
            arrays = {}
            metadata = None
            for arm, path in paths.items():
                if arm == "original": keys = ("pred_phaseformer_no_pcgrad_raw", "prediction_raw")
                elif arm == "exact": keys = ("pred_exact_pcgrad_raw",)
                elif arm == "full": keys = ("pred_scpcr_candidate_raw",)
                else: keys = ("pred_full_no_pcgrad_raw", "pred_scpcr_full_no_pcgrad_raw", "prediction_raw")
                pred, meta = load_arm(path, keys)
                if metadata is None: metadata = meta
                else:
                    checks = {name: np.array_equal(metadata[name], meta[name]) for name in metadata if name in meta}
                    if not all(checks.values()): raise RuntimeError(f"lineage mismatch in {key}/{arm}: {checks}")
                arrays[arm] = pred
                hashes[str(path.relative_to(ROOT))] = sha256(path)
            truth = metadata["truth_raw"]
            row = {"city": city, "seed": seed, "fold": key, "origins": int(truth.shape[0]), "horizon": int(truth.shape[1]), "sites": int(truth.shape[2]), "lineage_ok": True}
            for arm, pred in arrays.items():
                row[f"{arm}_rmse"], row[f"{arm}_mae"] = score(pred, truth)
            rows.append(row)
    if len(rows) != 18: raise RuntimeError(f"expected 18 folds, found {len(rows)}")
    with (OUT / "factorial_fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    rng = np.random.default_rng(RNG_SEED)
    comparisons = {}
    for left, right, label in (("exact", "full", "exact_vs_full"), ("original", "exact", "original_vs_exact"), ("original", "full_no_pcgrad", "original_vs_full_no_pcgrad"), ("full_no_pcgrad", "full", "full_no_pcgrad_vs_full")):
        comparisons[label] = {metric: bootstrap(rows, left, right, metric, rng) for metric in ("rmse", "mae")}
    # Difference-in-differences: (Full - Full-no-PCGrad) - (Exact - Original).
    interaction = {}
    for metric in ("rmse", "mae"):
        values = np.asarray([(r[f"full_{metric}"] - r[f"full_no_pcgrad_{metric}"]) - (r[f"exact_{metric}"] - r[f"original_{metric}"]) for r in rows], dtype=np.float64)
        ids = rng.integers(0, len(values), size=(REPLICATES, len(values)))
        boot = values[ids].mean(axis=1)
        interaction[metric] = {"point": float(values.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "unit": "paired city x seed fold", "interpretation": "PCGrad x relation interaction; negative means PCGrad contributes more when relation is enabled"}
    payload = {"status": "completed", "experiment": "formal paired Exact-PCGrad versus Full and 2x2 factorial interaction", "arms": {"original": "relation off, PCGrad off", "exact": "relation off, PCGrad on", "full_no_pcgrad": "relation on, PCGrad off", "full": "relation on, PCGrad on"}, "folds": 18, "cities": list(CITIES), "seeds": list(SEEDS), "protocol_id": "scpcr-matched-baseline-20261023", "retrospective": True, "comparisons": comparisons, "interaction": interaction, "input_hashes": hashes, "claim_boundary": "conditional on six observed cities, three fixed seeds, frozen checkpoints and retrospective origins; not a causal optimizer theorem"}
    (OUT / "factorial_ci.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Formal paired comparison and 2x2 interaction", "", "Status: **completed**; all four arms have 18 aligned city×seed folds and 10,000 paired bootstrap replicates.", "", "The primary comparison remains Original PhaseFormer versus Full SCPRC. Exact-PCGrad is a mechanism control. Full-no-PCGrad closes the factorial design so the relation effect, PCGrad effect and their interaction are separately estimable.", "", "## Paired comparisons", "", "| Comparison | Metric | Left | Right | Right−Left | 95% CI | Relative change | Better folds |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, items in comparisons.items():
        for metric, item in items.items():
            lines.append(f"| {label} | {metric.upper()} | {item['left_point']:.6f} | {item['right_point']:.6f} | {item['difference_right_minus_left']:+.6f} | [{item['difference_ci95'][0]:+.6f}, {item['difference_ci95'][1]:+.6f}] | {item['relative_change_percent']:+.3f}% | {item['right_better_folds']}/{item['folds']} |")
    lines += ["", "## Factorial interaction", "", "| Metric | Difference-in-differences | 95% CI |", "|---|---:|---:|"]
    for metric, item in interaction.items(): lines.append(f"| {metric.upper()} | {item['point']:+.6f} | [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] |")
    lines += ["", "Negative differences favor the right-hand model in pairwise rows. The interaction is a descriptive factorial attribution under the locked protocol; it does not establish that PCGrad is a universally causal improvement.", "", "Machine-readable results: `factorial_ci.json`; fold-level values: `factorial_fold_metrics.csv`."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "folds": len(rows), "comparisons": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

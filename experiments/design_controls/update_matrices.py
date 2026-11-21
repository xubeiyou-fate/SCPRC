#!/usr/bin/env python3
"""Render/apply a gated 2026-11-17 living-matrix extension.

The parent's no-argument invocation backs up both original Markdown files and
appends/replaces ONLY our marked section. The original freeze headers and
historical rows remain byte-identical.
Requires a full independent audit plus four complete 18-fold analysis summaries,
fold CSV values matching the audit, and paired energy intervals for every new
control against the same-batch Full reference. Never trains or imports models.

Example (after evaluate_controls.py and audit_completed_runs.py finish):
  python -B update_matrices.py --dry-run
  python -B update_matrices.py
Defaults use analysis/summaries.csv, per_fold.csv, uncertainty.csv. Unknown
CSV schemas fail closed; --columns-map JSON maps canonical names to CSV headers
using summary/fold/paired dictionaries, and optional model_aliases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("design_controls_artifact_audit", HERE / "audit_completed_runs.py")
auditlib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditlib)
ROOT, PROJECT = auditlib.ROOT, auditlib.PROJECT
VARIANTS, FOLDS = auditlib.VARIANTS, auditlib.FOLDS
ENERGY = auditlib.ENERGY_METRICS
METRICS = ("rmse", "mae", *ENERGY)
MODES = ("raw", "nonnegative_clip")
MATRICES = ("MAIN_EXPERIMENT_MATRIX.md", "MAIN_EXPERIMENT_MATRIX_ENERGY_20261104.md")
START = "<!-- BEGIN SUPPLEMENTARY_DESIGN_CONTROLS_20261117 -->"
END = "<!-- END SUPPLEMENTARY_DESIGN_CONTROLS_20261117 -->"
ROWS = {"phase_shared_strict": ("A4 / EV11", "Strict shared phase-relation operator"),
        "stationwise_matched": ("A5 / EV12", "Parameter-matched stationwise control"),
        "concat_project": ("A6 / EV13", "Concatenate-and-project fusion control")}


def require(condition, message):
    auditlib.require(condition, message)


def number(value, name):
    result = float(value)
    require(math.isfinite(result), f"nonfinite {name}: {value}")
    return result


def close(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f"{message}: {a} != {b}")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def value(row, name, aliases=(), mapping=None, optional=False):
    keys = ((mapping or {})[name],) if name in (mapping or {}) else (name, *aliases)
    found = [row[k] for k in keys if k in row and row[k] not in ("", None)]
    if not found and optional:
        return None
    require(found, f"missing CSV {name}; available={list(row)}; use --columns-map")
    return found[0]


def model_name(name, mapping):
    return mapping.get("model_aliases", {}).get(name, name)


def mode_name(name):
    names = {"raw": "raw", "clipped": "nonnegative_clip", "clip_nonnegative": "nonnegative_clip",
             "nonnegative": "nonnegative_clip", "nonnegative_clip": "nonnegative_clip"}
    require(name in names, f"unknown scoring postprocess {name}")
    return names[name]


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), f"missing/duplicate CSV headers: {path}")
        rows = list(reader)
    require(rows, f"empty analysis CSV: {path}")
    return rows


def choose_csv(root, explicit, names, description):
    if explicit:
        path = explicit if explicit.is_absolute() else root / explicit
        require(path.is_file(), f"missing {description}: {path}")
        return path.resolve()
    candidates = {p.resolve() for name in names for p in (root / "analysis").rglob(name) if p.is_file()}
    require(len(candidates) == 1, f"need one {description}; found {sorted(map(str, candidates))}; specify its CSV path")
    return candidates.pop()


def validate_audit(report, root, project):
    require(report.get("schema_version") == auditlib.SCHEMA_VERSION, "unsupported/missing independent audit schema")
    require(report.get("full_audit_pass") is True and report.get("status") == "passed" and not report.get("errors"),
            "full independent artifact audit did not pass; matrices will not be touched")
    require(Path(report.get("root", "")).resolve() == root and Path(report.get("project_root", "")).resolve() == project, "audit belongs to another root/project")
    require(report.get("expected_runs") == report.get("discovered_runs") == report.get("passed_runs") == 72, "audit must certify exactly 72 runs")
    require(report.get("expected_folds") == report.get("audited_folds") == 18, "audit must certify all 18 folds")
    summaries = report.get("model_summaries", {})
    require(set(summaries) == set(VARIANTS) and all(s["fold_count"] == 18 for s in summaries.values()), "four audited 18-fold summaries are required")
    expected = {(city, seed, variant) for city, seed in FOLDS for variant in VARIANTS}
    runs = report.get("runs", [])
    actual = [(r["city"], r["seed"], r["variant"]) for r in runs]
    require(len(actual) == 72 and set(actual) == expected and all(r.get("passed") is True for r in runs), "audit run identities incomplete/duplicate/failed")
    fold_records = report.get("folds", [])
    require(len(fold_records) == 18 and {(f["city"], f["seed"]) for f in fold_records} == set(FOLDS) and
            all(f.get("four_variant_initialization_and_sampling_pass") is True for f in fold_records), "initialization/sampler or alignment audit incomplete")
    # Rehash EVERY referenced input to prevent a stale pass authorizing new artifacts.
    files = {}
    def visit(item):
        if isinstance(item, dict):
            if {"path", "sha256", "size_bytes"} <= set(item):
                files[item["path"]] = item["sha256"]
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(report)
    require(files, "audit contains no verifiable artifact manifest")
    for path, digest in files.items():
        require(auditlib.sha256_file(Path(path)) == digest, f"stale audit: artifact changed {path}; rerun audit")
    require(report.get("auditor", {}).get("sha256") == auditlib.sha256_file(HERE / "audit_completed_runs.py"), "audit used a different auditor version")
    # Cross-check stored summary values against all 18 actual audit metric rows.
    indexed = {(r["city"], r["seed"], r["variant"]): r for r in runs}
    for variant in VARIANTS:
        for mode in MODES:
            for metric in METRICS:
                mean = sum(indexed[(c, s, variant)]["prediction"]["metrics"][mode][metric] for c, s in FOLDS) / 18
                close(summaries[variant]["metrics"][mode][metric], mean, "audit summary does not equal 18 fold values")
    return indexed


def parse_metrics(rows, kind, mappings):
    mapping = mappings.get(kind, {})
    parsed = {}
    for row in rows:
        model = model_name(value(row, "model", ("variant",), mapping), mappings)
        if model not in VARIANTS:
            continue
        city = value(row, "city", ("target_city", "target"), mapping, optional=kind == "summary")
        if kind == "summary" and city and city.upper() not in ("ALL", "ALL_18_FOLDS", "EQUAL_FOLD_MEAN", "OVERALL", "ALL_CITIES"):
            continue
        mode = mode_name(value(row, "postprocess", ("prediction_rule", "mode"), mapping))
        if kind == "summary":
            count = number(value(row, "n_folds", ("fold_count", "folds"), mapping), "summary fold count")
            require(count == 18, f"{model}/{mode} summary is not complete: {count} folds")
            key = (model, mode)
        else:
            city = city.upper()
            seed = number(value(row, "seed", (), mapping), "fold seed")
            require(seed.is_integer() and (city, int(seed)) in FOLDS, f"unexpected analysis fold {city}/{seed}")
            key = (model, mode, city, int(seed))
        metric_name = value(row, "metric", (), mapping, optional=True)
        if metric_name:
            metric_name = {"station_rmse": "rmse", "station_mae": "mae"}.get(metric_name, metric_name)
            if metric_name not in METRICS:
                continue
            result = parsed.setdefault(key, {})
            require(metric_name not in result, f"duplicate {kind} metric {key}/{metric_name}")
            result[metric_name] = number(value(row, "value", ("mean", "equal_fold_mean", "metric_value"), mapping), metric_name)
        else:
            require(key not in parsed, f"duplicate {kind} identity {key}")
            parsed[key] = {m: number(value(row, m, (m + "_mean", "equal_fold_mean_" + m, "station_" + m), mapping), m) for m in METRICS}
    expected = ({(v, mode) for v in VARIANTS for mode in MODES} if kind == "summary" else
                {(v, mode, c, s) for v in VARIANTS for mode in MODES for c, s in FOLDS})
    require(set(parsed) == expected and all(set(r) == set(METRICS) for r in parsed.values()),
            f"{kind} analysis must cover four models x both modes x all six metrics, all 18 folds; missing identities={len(expected - set(parsed))}")
    return parsed


def validate_analysis(summaries, folds, audited):
    for (variant, mode, city, seed), metrics in folds.items():
        for metric, actual in metrics.items():
            close(actual, audited[(city, seed, variant)]["prediction"]["metrics"][mode][metric],
                  f"analysis/audit mismatch {city}/{seed}/{variant}/{mode}/{metric}")
    for (variant, mode), metrics in summaries.items():
        for metric, actual in metrics.items():
            mean = sum(folds[(variant, mode, c, s)][metric] for c, s in FOLDS) / 18
            close(actual, mean, f"summary/fold mean mismatch {variant}/{mode}/{metric}")


def paired_rows(rows, summaries, mappings):
    mapping = mappings.get("paired", {})
    result, seen = [], set()
    for row in rows:
        if "control" in row and "reference" in row and "left_model" not in mapping:
            left, right = (model_name(row[k], mappings) for k in ("control", "reference"))
        else:
            left = model_name(value(row, "left_model", ("left", "reference", "reference_model"), mapping), mappings)
            right = model_name(value(row, "right_model", ("right", "model", "variant"), mapping), mappings)
        if "full_same_run" not in (left, right) or not ({left, right} - {"full_same_run"}) <= set(ROWS) or left == right:
            continue
        metric = value(row, "metric", (), mapping)
        metric = {"station_rmse": "rmse", "station_mae": "mae"}.get(metric, metric)
        if metric not in METRICS:
            continue
        mode = mode_name(value(row, "postprocess", ("mode",), mapping))
        method = value(row, "interval_method", ("method",), mapping)
        definition = value(row, "difference_definition", ("delta_definition",), mapping, optional=True)
        if definition is None and "difference_control_minus_full" in row:
            definition = "control minus Full; positive favors Full"
        require(definition, "paired difference must have an explicit direction")
        require(number(value(row, "n_folds", ("folds", "fold_count"), mapping), "paired folds") == 18, "paired interval missing 18 folds")
        require(number(value(row, "n_cities", ("cities",), mapping), "paired cities") == 6, "paired interval missing all six cities")
        mean = number(value(row, "mean_difference", ("difference_control_minus_full", "difference", "delta", "estimate"), mapping), "paired difference")
        low = number(value(row, "ci95_low", ("ci_low", "ci_lower"), mapping), "CI low")
        high = number(value(row, "ci95_high", ("ci_high", "ci_upper"), mapping), "CI high")
        require(low <= high, "reversed paired interval")
        compact = definition.lower().replace("_", " ").replace("−", "-").replace("–", "-")
        if "left" in compact and "right" in compact and ("minus" in compact or "-" in compact):
            sign = 1 if compact.index("left") < compact.index("right") else -1
            expected = sign * (summaries[(left, mode)][metric] - summaries[(right, mode)][metric])
        elif "control" in compact and "full" in compact and ("minus" in compact or "-" in compact):
            control = next(v for v in (left, right) if v != "full_same_run")
            sign = 1 if compact.index("control") < compact.index("full") else -1
            expected = sign * (summaries[(control, mode)][metric] - summaries[("full_same_run", mode)][metric])
        else:
            raise ValueError(f"unrecognized paired difference definition: {definition}; retain an explicit left-minus-right or control-minus-Full definition")
        close(mean, expected, f"paired effect does not match fold means: {left}/{right}/{metric}")
        identity = (left, right, mode, metric, method)
        require(identity not in seen, f"duplicate paired interval {identity}")
        seen.add(identity)
        replicates = number(value(row, "bootstrap_replicates", ("replicates",), mapping), "bootstrap replicates")
        require(replicates.is_integer() and replicates > 0, "invalid bootstrap replicate count")
        result.append({"left": left, "right": right, "postprocess": mode, "metric": metric,
                       "difference": mean, "ci_low": low, "ci_high": high, "method": method,
                       "definition": definition, "n_folds": 18,
                       "replicates": int(replicates),
                       "multiplicity": value(row, "multiplicity", ("interpretation",), mapping, optional=True)})
    covered = {(next(v for v in (r["left"], r["right"]) if v != "full_same_run"), r["postprocess"], r["metric"]) for r in result}
    expected = {(v, mode, metric) for v in ROWS for mode in MODES for metric in ENERGY}
    require(expected <= covered, f"paired energy evidence incomplete: missing {sorted(expected - covered)}")
    for method in ("six_city_cluster_bootstrap_all_three_seeds", "paired_168_origin_noncircular_moving_block_fixed_cities_seeds"):
        supported = {(next(v for v in (r["left"], r["right"]) if v != "full_same_run"), r["postprocess"], r["metric"]) for r in result if r["method"] == method}
        require(expected <= supported, f"missing complete {method} energy sensitivity evidence")
    return sorted(result, key=lambda r: (r["postprocess"], r["left"], r["right"], r["metric"], r["method"]))


def reference(path, project):
    try:
        return str(path.relative_to(project / "paper2"))
    except ValueError:
        return str(path)


def render(summaries, paired, report, sources, project):
    lines = [START, "## 2026-11-17 Supplementary Design Controls (Living Extension)", "",
        "This living-matrix extension preserves dated run identities and archived values. New evidence below updates the mechanism interpretation; older conclusions must be read with these controls. "
        "Original PhaseFormer (no-PCGrad) remains the primary baseline; Exact-PCGrad remains a mechanism/attribution control.", "",
        "Completion is gated on an independent 72/72 artifact audit: four models × six cities × seeds 13/17/29, "
        "18 folds each, 2,048 source minibatches / 8,192 window exposures / 410 canonical FP32 PCGrad updates. "
        "Truth, all origins 2928–3648, station order, checkpoint tensors, parameter counts, source/file hashes, "
        "shared backbone initialization and identical sampler traces pass; Full/strict-phase initial all-state hashes match.", "",
        "`full_same_run` is the **same-batch Full reference (18 folds)**. Its new-run identity remains separate from "
        "canonical Full SCPRC; these rows do not replace canonical Full or change the primary comparison.", "",
        "| ID | New control | Status | Paired reference |", "|---|---|---|---|"]
    for variant, (identifier, description) in ROWS.items():
        lines.append(f"| {identifier} | `{variant}` — {description} | Complete, audited 18/18 | `full_same_run`, 18/18 |")
    lines += ["", "### Actual Equal-Fold Metrics", "",
        "All values come from the validated analysis CSV, arithmetic mean over the same 18 folds. "
        "RMSE/MAE are site-level native-unit errors; Peak/Ramp/Total/Timing use city-summed trajectories. "
        "Raw and nonnegative-clipped forecasts are separate estimands; lower error is better.", "",
        "| Model | Scoring | Parameters | RMSE | MAE | Peak MAE | Ramp MAE | 24 h Total MAE | Timing MAE (h) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for model in VARIANTS:
            numbers = " | ".join(f"{summaries[(model, mode)][m]:.6f}" for m in METRICS)
            lines.append(f"| `{model}` | {mode} | {auditlib.PARAMETERS[model]} | {numbers} |")
    lines += ["", "### Paired Energy Trade-offs and Uncertainty", "",
        "The table retains every available paired interval for all three new controls, both scoring rules and all four "
        "energy endpoints, without selecting favorable outcomes. Differences and their directions are copied from the analysis "
        "and checked against the equal-fold means; interval methods and definitions are retained verbatim.", "",
        "| Left model | Right model | Scoring | Metric | Difference | 95% interval | Method | Difference definition | Bootstrap replicates |",
        "|---|---|---|---|---:|---|---|---|---:|"]
    for r in paired:
        lines.append(f"| `{r['left']}` | `{r['right']}` | {r['postprocess']} | {r['metric']} | {r['difference']:+.6f} | "
                     f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {cell(r['method'])} | {cell(r['definition'])} | {cell(r['replicates'] or 'not recorded')} |")
    lines += ["", "Each interval uses 18 paired city×seed folds conditional on six cities, three fixed seeds, shared source data "
        "and retrospective August 2023 outcomes. Folds/origins are not 18 independent cities or independent retraining populations. "
        "Report the supplied city-cluster and fixed-model time-block sensitivity methods with their stated scopes. "
        "An interval spanning zero is inconclusive for direction under that method. These exploratory endpoint-wise analyses "
        "do not establish joint superiority, mechanism necessity, unseen-city generalization, future-period performance or retraining uncertainty. "
        "Native-volume energy errors do not establish tariff savings or dispatch/reliability benefits.", ""]
    multiplicity = sorted({r["multiplicity"] for r in paired if r["multiplicity"]})
    if multiplicity:
        lines += ["Analysis multiplicity statements: " + "; ".join(cell(m) for m in multiplicity) + ".", ""]
    lines += ["### Evidence Lineage", "", f"Audit timestamp: `{report['created_at']}`.", ""]
    for label, record in sources.items():
        lines.append(f"- {label}: `{reference(Path(record['path']), project)}`; SHA256 `{record['sha256']}`.")
    lines += ["", END]
    return "\n".join(lines) + "\n"


def replace_section(original, section):
    require(original.count(START) == original.count(END) and original.count(START) <= 1, "unbalanced/duplicate extension markers")
    if START in original:
        start, end = original.index(START), original.index(END) + len(END)
        require(start < end - len(END), "reversed extension markers")
        return original[:start] + section.rstrip("\n") + original[end:]
    return original + ("" if original.endswith("\n\n") else "\n" if original.endswith("\n") else "\n\n") + section


def atomic_bytes(path, content, mode=None):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def integrate_overview(text, name):
    """Put the new controls in the existing overview table, not just an appendix."""
    main = name == 'MAIN_EXPERIMENT_MATRIX.md'
    heading = '## Full Experiment Matrix' if main else '## Energy-value matrix'
    begin = text.index(heading)
    end = text.find('\n## ', begin + len(heading))
    if end < 0: end = len(text)
    block = text[begin:end]
    additions = []
    for variant, (identifier, description) in ROWS.items():
        if main:
            row = f'| {identifier} | {description} | Same-batch Full vs `{variant}`; 18 paired folds | Complete; independent audit passed | Mechanism and energy attribution |'
        else:
            ev = identifier.split(' / ')[-1]
            row = f'| {ev} | {description} | Same-batch Full vs `{variant}` | Complete, 18/18 | See audited 2026-11-17 design-control results below | Mechanism and energy attribution |'
        if row not in block: additions.append(row)
    if additions:
        lines=block.splitlines();last=max(i for i,line in enumerate(lines) if line.startswith('|'))
        lines[last+1:last+1]=additions;block='\n'.join(lines)+'\n'
        text=text[:begin]+block+text[end:]
    if main:
        text=text.replace('# SCPRC Paper Matrix Freeze','# SCPRC Paper Experiment Matrix',1)
        text=text.replace('Freeze ID: `SCPRC-PAPER-20261031-v2`','Historical freeze ID: `SCPRC-PAPER-20261031-v2`',1)
        text=text.replace('Live status checked on 2026-11-01:', 'Historical status checked on 2026-11-01:',1)
    else:
        text=text.replace('Strict phase-conditioning control with all gates','Daily representation/relation replacement with all gates; EV11 isolates weight sharing')
        text=text.replace('Completed diagnostics support a conditional phase/peak benefit: Full lowers',
                          'The existing Daily replacement contrast changes both Q/K representation and relation sharing; EV11 below isolates normalized relation-weight sharing. Existing diagnostics show that Full lowers')
    note='**2026-11-17 matrix update:** Three supplementary design controls are complete and integrated into the overview table and audited results below. The same-batch Full reference has its own run identity; historical canonical scores are not silently relabeled. Read the [results and conclusions](experiments/supplementary_design_controls_20261117/RESULTS_AND_CONCLUSIONS_ZH.md) and [72-run independent audit](experiments/supplementary_design_controls_20261117/audit/completed_runs_audit.json).'
    if note not in text:
        first=text.index('\n');text=text[:first+1]+'\n'+note+'\n'+text[first+1:]
    return text


def apply_sections(root, project, section, sources):
    # Capture all original content and validate both targets before any mutation.
    originals, updates = {}, {}
    for name in MATRICES:
        path = project / "paper2" / name
        originals[path] = path.read_bytes()
        text = replace_section(originals[path].decode("utf-8"), section)
        text = integrate_overview(text, name)
        updates[path] = text.encode("utf-8")
    if all(updates[p] == originals[p] for p in originals):
        return {"status": "unchanged", "changed_files": []}
    # Unique immutable backup directory includes the exact originals and hashes.
    stamp = datetime.now(timezone.utc).strftime("matrix_backup_%Y%m%dT%H%M%S.%fZ")
    backup = root / "code_snapshots" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "originals": [], "evidence": sources}
    for path, original in originals.items():
        saved = backup / path.name
        saved.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        require(auditlib.sha256_file(saved) == digest, f"backup verification failed: {saved}")
        manifest["originals"].append({"path": str(path), "backup": str(saved), "sha256": digest,
                                      "size_bytes": len(original)})
    auditlib.atomic_json(backup / "originals_sha256.json", manifest)
    for path, original in originals.items():
        require(path.read_bytes() == original, f"matrix changed concurrently before write: {path}")
    written = []
    try:
        for path, content in updates.items():
            require(path.read_bytes() == originals[path], f"matrix changed concurrently: {path}")
            if content != originals[path]:
                atomic_bytes(path, content, path.stat().st_mode & 0o777)
                written.append(path)
                require(path.read_bytes() == content, f"matrix verification failed: {path}")
    except Exception:
        # Restore only files still containing exactly our version. Never clobber
        # a third party's intervening write during rollback.
        for path in reversed(written):
            if path.read_bytes() == updates[path]:
                atomic_bytes(path, originals[path], path.stat().st_mode & 0o777)
        raise
    manifest["updated"] = [{"path": str(p), "sha256": auditlib.sha256_file(p)} for p in written]
    auditlib.atomic_json(backup / "integration_receipt.json", manifest)
    return {"status": "applied", "changed_files": list(map(str, written)), "backup_directory": str(backup)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--project-root", type=Path, default=PROJECT)
    p.add_argument("--audit", type=Path, help="default ROOT/audit/completed_runs_audit.json")
    p.add_argument("--summary-csv", type=Path)
    p.add_argument("--fold-csv", type=Path)
    p.add_argument("--paired-csv", type=Path)
    p.add_argument("--columns-map", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="back up and edit both main matrices (default for parent no-argument invocation)")
    mode.add_argument("--dry-run", action="store_true", help="validate all gates, preview stdout, write nothing")
    args = p.parse_args(argv)
    root, project = args.root.resolve(), args.project_root.resolve()
    audit_path = (args.audit if args.audit and args.audit.is_absolute() else root / args.audit if args.audit else root / "audit/completed_runs_audit.json")
    try:
        audit_before = auditlib.file_record(audit_path)
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        audited = validate_audit(report, root, project)
        mappings = json.loads(args.columns_map.read_text()) if args.columns_map else {}
        paths = {"summary": choose_csv(root, args.summary_csv, ("summaries.csv", "model_summary.csv", "summary.csv", "equal_fold_summary.csv", "city_and_equal_fold_summary.csv"), "summary analysis"),
                 "fold": choose_csv(root, args.fold_csv, ("per_fold.csv", "fold_metrics.csv", "metrics_by_fold.csv", "fold_level_metrics.csv"), "fold analysis"),
                 "paired": choose_csv(root, args.paired_csv, ("uncertainty.csv", "paired_uncertainty.csv", "paired_energy_uncertainty.csv"), "paired uncertainty analysis")}
        sources = {"independent audit": audit_before, **{name + " CSV": auditlib.file_record(path) for name, path in paths.items()},
                   "matrix integration script": auditlib.file_record(Path(__file__))}
        manifest_path = root / "analysis/manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            require(manifest.get("status") == "complete" and manifest.get("evaluation_completed") is True and
                    manifest.get("evaluated_runs") == 72, "analysis manifest does not certify 72 completed evaluations")
            outputs = {str(Path(path).resolve()): digest for path, digest in manifest.get("output_sha256", {}).items()}
            for name, path in paths.items():
                require(outputs.get(str(path)) == sources[name + " CSV"]["sha256"], f"analysis manifest/{name} CSV hash mismatch")
            sources["analysis manifest"] = auditlib.file_record(manifest_path)
        summaries = parse_metrics(csv_rows(paths["summary"]), "summary", mappings)
        folds = parse_metrics(csv_rows(paths["fold"]), "fold", mappings)
        validate_analysis(summaries, folds, audited)
        paired = paired_rows(csv_rows(paths["paired"]), summaries, mappings)
        section = render(summaries, paired, report, sources, project)
        for record in sources.values():
            require(auditlib.sha256_file(Path(record["path"])) == record["sha256"], f"evidence changed during integration: {record['path']}")
        if not args.dry_run:
            print(json.dumps(apply_sections(root, project, section, sources), indent=2))
        else:
            print(section, end="")
            print("Preview only: no matrices or backups written. Parent no-argument invocation applies after all gates pass.", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Integration refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

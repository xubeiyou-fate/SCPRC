#!/usr/bin/env python3
"""Freeze a label-isolated Shanghai MP-EVData Q4 evaluation panel.

Only Jan-Sep 2024 history is read here. Q4 labels are opened by the runner
after it has frozen the source-only release decision and saved both models.
Battery-swap event counts (A7/A8) are excluded because they are not kW load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
DATA_DIR = ROOT / "scprc/data_adapters/public/mp_evdata_figshare_29882366"
RAW = DATA_DIR / "station-level load Profile 1h.xlsx"
OUT = ROOT / "scprc/training/mp_evdata_scpcr_external_20261021"
PANEL = OUT / "mp_evdata_history_prefix.npz"
FREEZE = OUT / "protocol_freeze.json"
SITES = ("A1", "A2", "A3", "A4", "A5", "A6", "A9")
PREFIX_START = pd.Timestamp("2024-01-01 00:00:00")
TEST_START = pd.Timestamp("2024-10-01 00:00:00")
TEST_END = pd.Timestamp("2025-01-01 00:00:00")
CONTEXT = 168
HORIZON = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if FREEZE.exists() or PANEL.exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation under {OUT}")
    if not RAW.is_file():
        raise FileNotFoundError(RAW)
    source_hash = sha256(RAW)
    metadata = json.loads((DATA_DIR / "figshare_metadata.json").read_text())
    file_meta = next(x for x in metadata["files"] if x["name"] == RAW.name)
    if int(file_meta["size"]) != RAW.stat().st_size or file_meta["computed_md5"] != "ef44311080071c9a2473c41e53043346":
        raise ValueError("downloaded hourly workbook does not match Figshare metadata")

    expected = pd.date_range(PREFIX_START, TEST_START, freq="h", inclusive="left")
    matrix = []
    stats = {}
    for site in SITES:
        # `nrows` ends before the Q4 test boundary; test labels are not parsed.
        frame = pd.read_excel(RAW, sheet_name=site, nrows=len(expected))
        if list(frame.columns) != ["datetime", "power"] or len(frame) != len(expected):
            raise ValueError(f"unexpected hourly schema/length for {site}: {frame.shape}, {list(frame.columns)}")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="coerce"))
        values = pd.to_numeric(frame["power"], errors="coerce").to_numpy(dtype=np.float64)
        if not timestamps.equals(expected) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"invalid prefix timestamps or load for {site}")
        matrix.append(values.astype(np.float32))
        stats[site] = {
            "prefix_mean_kw": float(values.mean()),
            "prefix_std_kw": float(values.std()),
            "prefix_zero_fraction": float(np.mean(values == 0)),
            "prefix_max_kw": float(values.max()),
        }
    values = np.stack(matrix, axis=1)
    if values.shape != (6576, len(SITES)):
        raise AssertionError(f"unexpected history shape {values.shape}")
    np.savez_compressed(PANEL,
                        history_timestamps=np.asarray(expected.astype(str), dtype=str),
                        site_ids=np.asarray(SITES, dtype=str),
                        history_kw=values.astype(np.float32),
                        target_prefix_end=np.asarray(len(expected), dtype=np.int64))
    starts = np.arange(len(expected), len(expected) + 24 * 92 - HORIZON + 1, dtype=np.int64)
    if len(starts) != 2185 or starts[-1] + HORIZON != len(expected) + 24 * 92:
        raise AssertionError("Q4 test origin arithmetic failed")
    record = {
        "status": "protocol_frozen_before_q4_read",
        "dataset": "MP-EVData 1-hour station load profiles",
        "paper": "Liu et al., Scientific Data (2026), DOI 10.1038/s41597-026-07273-5",
        "dataset_doi": "10.6084/m9.figshare.29882366.v3",
        "dataset_license": metadata["license"]["name"],
        "paper_license": "CC BY-NC-ND 4.0 (paper license; distinct from Figshare dataset license)",
        "raw_workbook": str(RAW.relative_to(ROOT)),
        "raw_size_bytes": RAW.stat().st_size,
        "raw_sha256": source_hash,
        "raw_figshare_md5": file_meta["computed_md5"],
        "prepared_history": str(PANEL.relative_to(ROOT)),
        "prepared_history_sha256": sha256(PANEL),
        "city": "Shanghai, China (separate from all six CHARGED source cities)",
        "site_ids": list(SITES),
        "site_types": {
            "A1-A2": "taxi demonstration charging stations",
            "A3-A4": "bus depots",
            "A5-A6": "residential charging stations",
            "A9": "heavy-duty truck logistics station",
        },
        "target": "hourly electrical power (kW), as published; no session reallocation",
        "excluded_sheets": {"A7": "battery-swap session count, not continuous kW", "A8": "battery-swap session count, not continuous kW", "A10": "starts 2024-09-27; excluded from full-year matched site panel"},
        "history_interval": f"{PREFIX_START} through {TEST_START - pd.Timedelta(hours=1)}",
        "external_test_interval": f"{TEST_START} through {TEST_END - pd.Timedelta(hours=1)}",
        "context_hours": CONTEXT,
        "horizon_hours": HORIZON,
        "rolling_origins": int(len(starts)),
        "rolling_update": "at every hourly forecast origin, only the immediately preceding 168 observed hours are input; later test observations may enter history after their timestamps",
        "target_training_or_scaling": False,
        "candidate_release": "frozen source-only inner-LOCO July gate and q90 support certificate; candidate only if both pass, otherwise exact PhaseFormer",
        "baselines": ["PhaseFormer exact", "SCPRC-v2.2-PCGrad candidate/strict release", "univariate source-only PatchTST", "persistence", "daily mean", "weekly naive"],
        "seeds": [13, 17, 29],
        "score_selection": False,
        "data_quality_prefix_summary": stats,
        "limitations": [
            "One external city with seven selected public station series supports an independent-city case study, not a population-level claim about most cities.",
            "This transfer experiment uses a harmonized 168-hour history and 24-hour hourly forecast to match CHARGED; it is not a scprc of MP-EVData paper baselines or its native next-day 15-minute benchmark setting.",
            "The seven stations have heterogeneous site types and load scales; report overall, equal-station macro, and station-specific metrics.",
            "Forecast origins overlap; use paired contiguous-block bootstrap within each fixed training seed."
        ],
    }
    FREEZE.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "history_shape": list(values.shape),
                      "test_origins": len(starts), "sites": list(SITES),
                      "freeze": str(FREEZE), "raw_sha256": source_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

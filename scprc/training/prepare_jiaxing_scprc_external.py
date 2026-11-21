#!/usr/bin/env python3
"""Freeze a session-to-hour external evaluation panel for Jiaxing EV data.

The source file is the public Figshare release linked by Scientific Data.
Because the public table contains transaction totals (not metered hourly load),
session kWh is allocated uniformly over each Start Time--End Time interval.
This preprocessing is fixed before running any forecasting model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
DATA_DIR = ROOT / "scprc/data_adapters/public/jiaxing_figshare_28182251"
RAW_ZIP = DATA_DIR / "Dataset.zip"
OUT_DIR = ROOT / "scprc/training/jiaxing_scpcr_external_20261021"
CSV_MEMBER = "Dataset/Charging_Data.csv"
START = pd.Timestamp("2020-01-01 00:00:00")
TEST_START = pd.Timestamp("2021-07-01 00:00:00")
TEST_END = pd.Timestamp("2021-10-01 00:00:00")  # exclusive; Q3 2021 is the locked test
HORIZON = 24
CONTEXT = 168
SOURCE_PROFILE_HOURS = 2928  # same 122-day prefix length as CHARGED Apr--Jul


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allocate_sessions(frame: pd.DataFrame, locations: tuple[str, ...], times: pd.DatetimeIndex):
    n_hours, n_sites = len(times), len(locations)
    energy = np.zeros((n_hours, n_sites), dtype=np.float64)
    duration = np.zeros_like(energy)
    loc_index = {name: i for i, name in enumerate(locations)}
    grid_start = times[0]
    grid_end = times[-1] + pd.Timedelta(hours=1)
    outside = 0
    zero_duration = 0

    # Work row-wise to keep the hour-overlap allocation transparent and exact.
    for loc, start, end, kwh_value in frame.itertuples(index=False, name=None):
        kwh = float(kwh_value)
        loc_i = loc_index[loc]
        if end < start:
            raise ValueError(f"negative session duration: {start} -- {end}")
        seconds = (end - start).total_seconds()
        if seconds <= 0:
            zero_duration += 1
            if grid_start <= start < grid_end:
                hour = int((start - grid_start).total_seconds() // 3600)
                energy[hour, loc_i] += kwh
            else:
                outside += 1
            continue
        left = max(start, grid_start)
        right = min(end, grid_end)
        if right <= left:
            outside += 1
            continue
        cursor = left
        while cursor < right:
            hour_start = cursor.floor("h")
            hour_end = hour_start + pd.Timedelta(hours=1)
            segment_end = min(right, hour_end)
            overlap = (segment_end - cursor).total_seconds()
            hour = int((hour_start - grid_start).total_seconds() // 3600)
            if 0 <= hour < n_hours:
                energy[hour, loc_i] += kwh * overlap / seconds
                duration[hour, loc_i] += overlap / 3600.0
            cursor = segment_end
    return energy, duration, {"sessions_outside_hourly_panel": outside,
                              "zero_duration_sessions": zero_duration}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = OUT_DIR / "protocol_freeze.json"
    prepared_path = OUT_DIR / "jiaxing_hourly_panel.npz"
    if freeze_path.exists() or prepared_path.exists():
        raise FileExistsError(f"frozen output already exists under {OUT_DIR}; do not overwrite")
    if not RAW_ZIP.is_file():
        raise FileNotFoundError(RAW_ZIP)

    with ZipFile(RAW_ZIP) as archive:
        with archive.open(CSV_MEMBER) as stream:
            raw = pd.read_csv(stream, encoding="gb18030", low_memory=False)
    required = {"Location Information", "Start Time", "End Time", "Transaction power/kwh"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    raw["_loc"] = raw["Location Information"].astype(str).str.strip()
    raw["_start"] = pd.to_datetime(raw["Start Time"].astype(str).str.strip(), errors="coerce")
    raw["_end"] = pd.to_datetime(raw["End Time"].astype(str).str.strip(), errors="coerce")
    raw["_kwh"] = pd.to_numeric(raw["Transaction power/kwh"], errors="coerce")
    if raw[["_loc", "_start", "_end", "_kwh"]].isna().any().any():
        raise ValueError("null/invalid location, timestamp or transaction energy")
    if (raw["_kwh"] < 0).any():
        raise ValueError("negative transaction energy found")
    # Do not materialize the test after 2021-09-30. The panel is fixed to 13
    # published station locations and uses only the first 3 test months.
    panel_mask = (raw["_end"] >= START) & (raw["_start"] < TEST_END)
    frame = raw.loc[panel_mask, ["_loc", "_start", "_end", "_kwh"]].copy()
    locations = tuple(sorted(frame["_loc"].unique()))
    if len(locations) != 13:
        raise ValueError(f"expected 13 published location groups, got {len(locations)}")
    times = pd.date_range(START, TEST_END, freq="h", inclusive="left")
    energy, duration, allocation_audit = allocate_sessions(frame, locations, times)
    if energy.shape[0] <= SOURCE_PROFILE_HOURS or not np.isfinite(energy).all() or not np.isfinite(duration).all():
        raise ValueError("invalid aggregated panel")
    if (energy < 0).any() or (duration < 0).any():
        raise ValueError("negative hourly values")

    # Prefix labels are forecast history only; test begins after the June 2021 prefix.
    prefix_end = int((TEST_START - START).total_seconds() // 3600)
    test_start = prefix_end
    test_end = len(times)
    starts = np.arange(test_start, test_end - HORIZON + 1, dtype=np.int64)
    if len(starts) != 2185 or starts[-1] + HORIZON != test_end:
        raise ValueError(f"unexpected locked test window count {len(starts)}")

    np.savez_compressed(
        prepared_path,
        times=np.asarray(times.astype(str), dtype=str),
        site_ids=np.asarray(locations, dtype=str),
        energy_kwh=energy.astype(np.float32),
        duration_hours=duration.astype(np.float32),
        target_prefix_end=np.asarray(prefix_end, dtype=np.int64),
        test_start=np.asarray(test_start, dtype=np.int64),
        test_end=np.asarray(test_end, dtype=np.int64),
        test_starts=starts,
    )
    digest = {
        "dataset_title": "A high-resolution electric vehicle charging transaction dataset with multidimensional features in China",
        "article_doi": "10.1038/s41597-025-04982-1",
        "figshare_doi": "10.6084/m9.figshare.28182251.v3",
        "license": "CC BY 4.0",
        "raw_file": str(RAW_ZIP.relative_to(ROOT)),
        "raw_file_size_bytes": RAW_ZIP.stat().st_size,
        "raw_file_md5": md5(RAW_ZIP),
        "raw_file_sha256": sha256(RAW_ZIP),
        "prepared_file": str(prepared_path.relative_to(ROOT)),
        "prepared_sha256": sha256(prepared_path),
        "protocol": {
            "location_nodes": 13,
            "time_zone": "Asia/Shanghai (naive local timestamps; no DST transitions)",
            "grid_start": str(START),
            "grid_end_exclusive": str(TEST_END),
            "target_history_prefix": f"{START} through {TEST_START - pd.Timedelta(hours=1)}",
            "primary_test": f"{TEST_START} through {TEST_END - pd.Timedelta(hours=1)}",
            "context_hours": CONTEXT,
            "horizon_hours": HORIZON,
            "forecast_origins": int(len(starts)),
            "source_training": "CHARGED v3 all six cities, source Apr-Jul only, 2048 optimizer updates per seed",
            "target_training_or_target_label_tuning": False,
            "target_scaler": "none; fixed CHARGED source-only scaler",
            "strict_release": "candidate only when the fixed source-support certificate passes; otherwise exact PhaseFormer",
            "seeds": [13, 17, 29],
            "features": "SCPRC/PhaseFormer: session-derived hourly kWh only; PatchTST control: same hourly kWh plus session-overlap duration-hours",
            "session_to_hour_rule": "split each transaction kWh in proportion to overlap seconds in each local clock hour of [Start Time, End Time); use the full elapsed session interval as a uniform-power proxy; zero-duration session energy assigned to its start hour",
            "unobserved_site_hours": "zero transaction energy in the public transaction table; this is an observed-session-derived proxy, not separately metered zero load",
            "scoring": "raw kWh RMSE/MAE, equal-location macro RMSE/MAE, 13-location aggregate RMSE/MAE, p95 rolling-window RMSE; paired contiguous 168-origin block bootstrap across seeds",
            "excluded_rows": "sessions outside the locked Jan 2020-Sep 2021 panel; no post-Sep 2021 labels are materialized",
            "score_based_selection": False,
            "gate_search_or_expert_pool": False,
        },
        "data_audit": {
            "raw_records_total": int(len(raw)),
            "records_overlapping_locked_panel": int(len(frame)),
            "first_session_start_in_panel": str(frame["_start"].min()),
            "last_session_end_in_panel": str(frame["_end"].max()),
            "allocated_energy_kwh": float(energy.sum()),
            "zero_transaction_records": int((frame["_kwh"] == 0).sum()),
            "allocation": allocation_audit,
            "location_names": list(locations),
        },
    }
    # Persist the lock before any model is trained or test score is computed.
    freeze_path.write_text(json.dumps(digest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "freeze": str(freeze_path),
                      "prepared": str(prepared_path), "origins": len(starts),
                      "locations": list(locations), "records": len(frame),
                      "energy_kwh": float(energy.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

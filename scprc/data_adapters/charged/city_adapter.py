"""Audited loader for the pinned CHARGED v3 hourly city archive.

The archive is the official Zenodo v3 ``Hourly.zip`` payload.  This module
keeps each city in its published local-hour grid and never mixes site columns
across cities.  It is deliberately separate from the third-party session
adapter: CHARGED is already released as hourly kWh/site matrices.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scprc.data_adapters.cross_city_adapter import CitySeries


ARCHIVE = Path(__file__).resolve().parent / "Hourly.zip"
DATASET_DOI = "10.5281/zenodo.15814263"
ARTICLE_DOI = "10.1038/s41597-025-05584-7"
LICENSE = "CC BY 4.0"
ARCHIVE_MD5 = "099a592653ca74fa1f1051e9c23d6456"
FIRST_HOUR = pd.Timestamp("2023-04-01 00:00:00")
LAST_HOUR = pd.Timestamp("2023-09-30 23:00:00")
CITY_CODES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")


def load_city(city_code: str, *, archive_path: Path = ARCHIVE,
              remove_zero: bool = False) -> CitySeries:
    """Load and validate one standard CHARGED hourly city matrix."""
    code = str(city_code).upper()
    if code not in CITY_CODES:
        raise ValueError(f"city_code must be one of {CITY_CODES}")
    member = f"{code}{'_remove_zero' if remove_zero else ''}/volume.csv"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member) as stream:
                frame = pd.read_csv(io.BytesIO(stream.read()))
        except KeyError as error:
            raise FileNotFoundError(f"missing CHARGED archive member: {member}") from error
    if frame.shape[1] < 2:
        raise ValueError(f"{member} must contain timestamp and site columns")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range(FIRST_HOUR, LAST_HOUR, freq="h")
    if not times.equals(expected):
        raise ValueError(
            f"{member} does not match the published 4,392-hour grid: "
            f"observed {len(times)} rows"
        )
    site_ids = tuple(str(value) for value in frame.columns[1:])
    if len(set(site_ids)) != len(site_ids) or not all(site_ids):
        raise ValueError(f"{member} has duplicate or empty site identifiers")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if values.shape != (len(expected), len(site_ids)):
        raise ValueError(f"{member} has inconsistent value shape")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{member} has missing, nonfinite, or negative values")
    return CitySeries(
        city_id=code,
        times=times,
        node_ids=site_ids,
        values=values,
        unit="kWh per site per local hour",
        source=f"Zenodo:{DATASET_DOI}:{member}",
        timezone="published local-hour grid",
        metadata={
            "dataset": "CHARGED",
            "dataset_doi": DATASET_DOI,
            "article_doi": ARTICLE_DOI,
            "license": LICENSE,
            "archive_member": member,
            "archive_md5": ARCHIVE_MD5,
            "remove_zero": bool(remove_zero),
            "preprocessing_note": "released series includes published anomaly correction/interpolation; see descriptor article",
        },
    )


def load_city_duration(city_code: str, *, archive_path: Path = ARCHIVE,
                       remove_zero: bool = False) -> CitySeries:
    """Load the leakage-safe CHARGED hourly session-duration matrix.

    Duration is aligned to the same timestamp/site grid as ``load_city``;
    mismatched columns, missing values, negatives, and archive grid drift are
    rejected before a caller can construct an auxiliary training target.
    """
    code = str(city_code).upper()
    if code not in CITY_CODES:
        raise ValueError(f"city_code must be one of {CITY_CODES}")
    member = f"{code}{'_remove_zero' if remove_zero else ''}/duration.csv"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member) as stream:
                frame = pd.read_csv(io.BytesIO(stream.read()))
        except KeyError as error:
            raise FileNotFoundError(f"missing CHARGED archive member: {member}") from error
    if frame.shape[1] < 2:
        raise ValueError(f"{member} must contain timestamp and site columns")
    times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
    expected = pd.date_range(FIRST_HOUR, LAST_HOUR, freq="h")
    if not times.equals(expected):
        raise ValueError(f"{member} does not match the published 4,392-hour grid")
    site_ids = tuple(str(value) for value in frame.columns[1:])
    if len(set(site_ids)) != len(site_ids) or not all(site_ids):
        raise ValueError(f"{member} has duplicate or empty site identifiers")
    volume = load_city(code, archive_path=archive_path, remove_zero=remove_zero)
    if site_ids != volume.node_ids:
        raise ValueError(f"{member} site columns do not match volume.csv order")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if values.shape != volume.values.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{member} has inconsistent, missing, nonfinite, or negative values")
    return CitySeries(
        city_id=code, times=times, node_ids=site_ids, values=values,
        unit="minutes per site per local hour", source=f"Zenodo:{DATASET_DOI}:{member}",
        timezone="published local-hour grid",
        metadata={"dataset": "CHARGED", "dataset_doi": DATASET_DOI,
                  "archive_member": member, "archive_md5": ARCHIVE_MD5,
                  "remove_zero": bool(remove_zero),
                  "paired_volume_source": volume.source},
    )


def load_city_tariff_features(city_code: str, *, archive_path: Path = ARCHIVE,
                              remove_zero: bool = False) -> np.ndarray:
    """Load leakage-safe hourly electricity/service price summaries.

    The released price matrices are aligned to the same local-hour grid as
    volume.  We retain only hourly cross-site summaries (mean, dispersion and
    active-site fraction) so the feature dimension is invariant across cities;
    no volume, duration, or validation label is read.
    """
    code = str(city_code).upper()
    if code not in CITY_CODES:
        raise ValueError(f"city_code must be one of {CITY_CODES}")
    matrices = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in ("e_price.csv", "s_price.csv"):
            member = f"{code}{'_remove_zero' if remove_zero else ''}/{name}"
            try:
                with archive.open(member) as stream:
                    frame = pd.read_csv(io.BytesIO(stream.read()))
            except KeyError as error:
                raise FileNotFoundError(f"missing CHARGED archive member: {member}") from error
            if frame.shape[1] < 2:
                raise ValueError(f"{member} must contain timestamp and site columns")
            times = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            expected = pd.date_range(FIRST_HOUR, LAST_HOUR, freq="h")
            if not times.equals(expected):
                raise ValueError(f"{member} does not match the published hourly grid")
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{member} has missing, nonfinite, or negative values")
            matrices.append(np.column_stack((
                values.mean(axis=1),
                values.std(axis=1),
                (values > 0).mean(axis=1),
            )))
    return np.concatenate(matrices, axis=1).astype(np.float32)


def audit_cities(*, city_codes: tuple[str, ...] = CITY_CODES,
                 archive_path: Path = ARCHIVE) -> list[dict[str, object]]:
    """Validate all selected standard city matrices without model access."""
    rows = []
    for code in city_codes:
        series = load_city(code, archive_path=archive_path)
        rows.append({
            "city": code,
            "hours": int(len(series.times)),
            "sites": int(len(series.node_ids)),
            "start": str(series.times[0]),
            "end": str(series.times[-1]),
            "finite_nonnegative": bool(np.isfinite(series.values).all() and (series.values >= 0).all()),
            "zero_fraction": float((series.values == 0).mean()),
            "source": series.source,
            "license": LICENSE,
        })
    return rows


def load_static_grid_assignment(city_code: str, *, archive_path: Path = ARCHIVE,
                                n_lat: int = 2, n_lon: int = 2) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a coordinate-only site→region matrix from ``sites.csv``.

    The rectangular grid is fixed from static latitude/longitude extents only;
    no volume, duration, activity, or future label is used.  Empty cells are
    removed and returned rows are one-hot region assignments.
    """
    code = str(city_code).upper()
    if code not in CITY_CODES or n_lat < 1 or n_lon < 1:
        raise ValueError("invalid city code or grid dimensions")
    member = f"{code}/sites.csv"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            sites = pd.read_csv(io.BytesIO(stream.read()))
    site_key = "site_id" if "site_id" in sites.columns else "site"
    required = {site_key, "longitude", "latitude"}
    if not required.issubset(sites.columns):
        raise ValueError(f"{member} is missing static coordinate columns")
    volume = load_city(code, archive_path=archive_path)
    sites[site_key] = sites[site_key].astype(str)
    lookup = sites.set_index(site_key)
    if set(volume.node_ids) != set(lookup.index.astype(str)):
        raise ValueError(f"{member} site IDs do not match {code}/volume.csv")
    ordered = lookup.loc[list(volume.node_ids)]
    lon = ordered["longitude"].to_numpy(dtype=float)
    lat = ordered["latitude"].to_numpy(dtype=float)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError(f"{member} contains non-finite coordinates")
    lon_edges = np.linspace(lon.min(), lon.max(), n_lon + 1)
    lat_edges = np.linspace(lat.min(), lat.max(), n_lat + 1)
    lon_bin = np.clip(np.digitize(lon, lon_edges[1:-1], right=False), 0, n_lon - 1)
    lat_bin = np.clip(np.digitize(lat, lat_edges[1:-1], right=False), 0, n_lat - 1)
    cell_ids = lat_bin * n_lon + lon_bin
    active = sorted(set(int(value) for value in cell_ids.tolist()))
    row_by_cell = {cell: row for row, cell in enumerate(active)}
    assignment = np.zeros((len(active), len(volume.node_ids)), dtype=np.float32)
    for col, cell in enumerate(cell_ids.tolist()):
        assignment[row_by_cell[int(cell)], col] = 1.0
    labels = tuple(f"grid_{cell}" for cell in active)
    return assignment, labels


def load_static_node_features(city_code: str, *, archive_path: Path = ARCHIVE
                              ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load leakage-safe site metadata aligned to ``volume.csv`` columns.

    Only coordinates, charger count, and static geometry are returned.  Fields
    such as ``total_volume`` and ``total_duration`` are deliberately excluded:
    they summarize charging observations and would leak target-city labels in a
    strict source-only fold.  Coordinates are centered/scaled within the city
    (an unsupervised geometric transform); count and area use log transforms
    followed by within-city robust median/IQR scaling.  The transform is
    computed from static metadata only and is therefore transductive geometry,
    not a target-label statistic.
    """
    code = str(city_code).upper()
    if code not in CITY_CODES:
        raise ValueError(f"city_code must be one of {CITY_CODES}")
    member = f"{code}/sites.csv"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            sites = pd.read_csv(io.BytesIO(stream.read()))
    site_key = "site_id" if "site_id" in sites.columns else "site"
    required = {site_key, "longitude", "latitude", "charger_num", "area", "perimeter"}
    if not required.issubset(sites.columns):
        raise ValueError(f"{member} is missing safe static feature columns")
    volume = load_city(code, archive_path=archive_path)
    sites[site_key] = sites[site_key].astype(str)
    ordered = sites.set_index(site_key).loc[list(volume.node_ids)]
    raw = np.column_stack([
        ordered["longitude"].to_numpy(dtype=np.float64),
        ordered["latitude"].to_numpy(dtype=np.float64),
        np.log1p(np.maximum(ordered["charger_num"].to_numpy(dtype=np.float64), 0.0)),
        np.log1p(np.maximum(ordered["area"].to_numpy(dtype=np.float64), 0.0)),
        np.log1p(np.maximum(ordered["perimeter"].to_numpy(dtype=np.float64), 0.0)),
    ])
    if not np.isfinite(raw).all():
        raise ValueError(f"{member} contains non-finite static features")
    median = np.median(raw, axis=0, keepdims=True)
    q75, q25 = np.percentile(raw, [75.0, 25.0], axis=0, keepdims=True)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    # A few released geometry fields are nearly constant in small cities;
    # clipping the robust z-score prevents one rare site from dominating the
    # shared encoder while keeping the transform label-free.
    features = np.clip((raw - median) / scale, -5.0, 5.0).astype(np.float32)
    return features, (
        "longitude_median_iqr", "latitude_median_iqr",
        "log_charger_median_iqr", "log_area_median_iqr",
        "log_perimeter_median_iqr",
    )


def load_static_capacity(city_code: str, *, archive_path: Path = ARCHIVE,
                         minimum: float = 1.0) -> np.ndarray:
    """Return leakage-safe per-site charger capacity aligned to ``volume.csv``.

    ``charger_num`` is static infrastructure metadata.  It is used only as a
    positive denominator for capacity-normalized utilization; no volume,
    duration, price, or target-derived site summary is consulted.
    """
    code = str(city_code).upper()
    if code not in CITY_CODES or minimum <= 0:
        raise ValueError("invalid city code or minimum capacity")
    member = f"{code}/sites.csv"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            sites = pd.read_csv(io.BytesIO(stream.read()))
    site_key = "site_id" if "site_id" in sites.columns else "site"
    if not {site_key, "charger_num"}.issubset(sites.columns):
        raise ValueError(f"{member} is missing charger_num")
    volume = load_city(code, archive_path=archive_path)
    sites[site_key] = sites[site_key].astype(str)
    ordered = sites.set_index(site_key).loc[list(volume.node_ids)]
    capacity = ordered["charger_num"].to_numpy(dtype=np.float64)
    if not np.isfinite(capacity).all() or np.any(capacity < 0):
        raise ValueError(f"{member} contains invalid charger_num")
    return np.maximum(capacity, float(minimum))


if __name__ == "__main__":
    import json
    print(json.dumps(audit_cities(), indent=2, ensure_ascii=False))


__all__ = [
    "ARCHIVE", "CITY_CODES", "load_city", "load_city_duration", "audit_cities",
    "load_static_grid_assignment", "load_static_node_features", "load_static_capacity",
]

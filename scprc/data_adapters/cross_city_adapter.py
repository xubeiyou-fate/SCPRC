"""Small, leakage-audited schema for pooled and leave-one-city-out studies.

This module deliberately does *not* discover or open CHARGED files.  Callers
must pass an already audited :class:`CitySeries` (or adapt the LOA-only
loader's in-memory ``LOASeries``).  That keeps the JHB/MEL/SPO final-city
boundary enforceable while allowing the cross-city runner to use one common
window/scaling contract.

The pooled representation is a tuple of per-city datasets rather than a
single stacked tensor: CHARGED cities have different numbers of sites.  A
model may therefore consume one city batch at a time while sharing parameters
and city metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


CONTEXT = 168
HORIZON = 24
Boundary = int | str | pd.Timestamp
BoundarySpec = Boundary | Mapping[str, Boundary]


@dataclass(frozen=True)
class CitySeries:
    """Validated hourly multivariate series for one city.

    ``values`` has shape ``[hours, nodes]`` and is kept in the source unit
    (normally kWh per site per hour).  No file path is accepted on purpose;
    source loading and final-city access policy remain outside this schema.
    """

    city_id: str
    times: pd.DatetimeIndex
    node_ids: tuple[str, ...]
    values: np.ndarray
    unit: str = "kWh per node per hour"
    source: str = ""
    timezone: str = "local-naive"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.city_id:
            raise ValueError("city_id must be nonempty")
        if not isinstance(self.times, pd.DatetimeIndex):
            raise TypeError("times must be a pandas DatetimeIndex")
        if len(self.times) < 2 or not self.times.is_monotonic_increasing:
            raise ValueError("times must contain at least two ordered timestamps")
        if not self.times.is_unique:
            raise ValueError("times must be unique")
        # Reject gaps and DST folds here.  A city adapter may normalize a
        # source to UTC before constructing CitySeries, but cannot silently
        # create a training window over a missing/duplicated hour.
        deltas = np.diff(self.times.asi8)
        if not np.all(deltas == pd.Timedelta(hours=1).value):
            raise ValueError("times must be a continuous hourly grid")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique and nonempty")
        values = np.asarray(self.values)
        if values.ndim != 2 or values.shape != (len(self.times), len(self.node_ids)):
            raise ValueError("values must have shape [hours, nodes]")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("values must be numeric")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("values must be finite and nonnegative")


@dataclass(frozen=True)
class CityScaler:
    """Per-node or broadcast scalar scaler fitted on an allowed prefix."""

    mean: np.ndarray
    std: np.ndarray
    fit_end_exclusive: int
    node_scale: np.ndarray | None = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if self.node_scale is not None:
            scale = np.asarray(self.node_scale, dtype=np.float64).reshape(-1)
            if array.shape[-1] != len(scale) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("node_scale does not match the node dimension")
            array = array / scale
        if self.mean.size != 1 and array.shape[-1] != len(self.mean):
            raise ValueError("node dimension does not match fitted scaler")
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
        if self.mean.size != 1 and array.shape[-1] != len(self.mean):
            raise ValueError("node dimension does not match fitted scaler")
        restored = array.astype(np.float64) * self.std + self.mean
        if self.node_scale is not None:
            scale = np.asarray(self.node_scale, dtype=np.float64).reshape(-1)
            if restored.shape[-1] != len(scale):
                raise ValueError("node_scale does not match the node dimension")
            restored = restored * scale
        return restored


def fit_city_scaler(values: np.ndarray, train_end: int) -> CityScaler:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not 0 < train_end <= len(values):
        raise ValueError("invalid values or train_end")
    prefix = values[:train_end]
    mean = prefix.mean(axis=0)
    std = prefix.std(axis=0)
    std[std < 1e-6] = 1.0
    return CityScaler(mean=mean, std=std, fit_end_exclusive=int(train_end))


def fit_source_global_scaler(
    cities: Mapping[str, CitySeries], source_city_ids: tuple[str, ...],
    train_end: BoundarySpec,
    *, node_scales: Mapping[str, np.ndarray] | None = None,
) -> CityScaler:
    """Fit one broadcastable scaler using source-city prefixes only.

    A length-one ``mean/std`` is intentional: it can transform cities with
    different station counts while ensuring that no target-city statistic is
    used to define the representation.  The returned ``fit_end_exclusive``
    records the largest source boundary and is metadata, not a target index.
    """
    if not source_city_ids or any(city_id not in cities for city_id in source_city_ids):
        raise ValueError("source_city_ids must name supplied cities")
    city_means: list[float] = []
    city_second_moments: list[float] = []
    max_end = 0
    for city_id in source_city_ids:
        end = _boundary(cities[city_id].times, train_end[city_id] if isinstance(train_end, Mapping) else train_end, "train_end")
        values = np.asarray(cities[city_id].values[:end], dtype=np.float64)
        if node_scales is not None:
            if city_id not in node_scales:
                raise ValueError(f"node_scales missing source city {city_id}")
            scale = np.asarray(node_scales[city_id], dtype=np.float64).reshape(-1)
            if scale.shape != (len(cities[city_id].node_ids),) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"node_scales[{city_id}] must be positive with one value per node")
            values = values / scale[None, :]
        city_means.append(float(values.mean()))
        city_second_moments.append(float(np.square(values).mean()))
        max_end = max(max_end, end)
    # Give each source city equal weight instead of letting a city with more
    # stations dominate the shared representation.
    mean_value = float(np.mean(city_means))
    second_value = float(np.mean(city_second_moments))
    mean = np.asarray([mean_value], dtype=np.float64)
    std_value = float(np.sqrt(max(second_value - mean_value ** 2, 0.0)))
    std = np.asarray([std_value if std_value >= 1e-6 else 1.0], dtype=np.float64)
    return CityScaler(mean=mean, std=std, fit_end_exclusive=max_end)


class CityWindows(Dataset):
    """Consecutive windows whose labels stay inside one split interval."""

    def __init__(self, scaled_values: np.ndarray, times: pd.DatetimeIndex,
                 label_start: int, label_end: int, *, context: int = CONTEXT,
                 horizon: int = HORIZON, city_id: str = "") -> None:
        values = np.asarray(scaled_values, dtype=np.float32)
        if values.ndim != 2 or len(values) != len(times):
            raise ValueError("scaled_values and times must agree")
        if context <= 0 or horizon <= 0:
            raise ValueError("context and horizon must be positive")
        if not 0 <= label_start < label_end <= len(times):
            raise ValueError("invalid label interval")
        first = max(int(label_start), int(context))
        last = int(label_end) - int(horizon)
        if last < first:
            raise ValueError("split has no complete context/label window")
        self.values = values
        self.times = times
        self.city_id = city_id
        self.context = int(context)
        self.horizon = int(horizon)
        self.label_start = int(label_start)
        self.label_end = int(label_end)
        self.target_starts = np.arange(first, last + 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.target_starts[index])
        x = self.values[start - self.context:start]
        y = self.values[start:start + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

    def label_times(self, index: int) -> pd.DatetimeIndex:
        start = int(self.target_starts[index])
        return self.times[start:start + self.horizon]


@dataclass(frozen=True)
class CityFold:
    city_id: str
    node_ids: tuple[str, ...]
    scaler: CityScaler
    train: CityWindows
    val: CityWindows
    context: int = CONTEXT
    horizon: int = HORIZON
    unit: str = "kWh per node per hour"
    source: str = ""


def _boundary(times: pd.DatetimeIndex, value: int | str | pd.Timestamp, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        index = int(value)
    else:
        stamp = pd.Timestamp(value)
        try:
            index = int(times.get_loc(stamp))
        except KeyError as error:
            # An exclusive end is often written as the first hour after a
            # file's last row (for example 2023-10-01 for a September file).
            # Accept that unambiguous boundary without synthesizing a row.
            if stamp == times[-1] + pd.Timedelta(hours=1):
                index = len(times)
            else:
                raise ValueError(f"{name} timestamp is not present in city grid: {stamp}") from error
    if not 0 < index <= len(times):
        raise ValueError(f"{name} must be in (0, len(times)]")
    return index


def prepare_city_fold(series: CitySeries, train_end: int | str | pd.Timestamp,
                      val_end: int | str | pd.Timestamp, *,
                      context: int = CONTEXT, horizon: int = HORIZON) -> CityFold:
    """Prepare one city without using validation values in its scaler."""
    train_i = _boundary(series.times, train_end, "train_end")
    val_i = _boundary(series.times, val_end, "val_end")
    if val_i <= train_i:
        raise ValueError("val_end must be after train_end")
    scaler = fit_city_scaler(series.values, train_i)
    scaled = scaler.transform(series.values[:val_i])
    times = series.times[:val_i]
    return CityFold(
        city_id=series.city_id,
        node_ids=series.node_ids,
        scaler=scaler,
        train=CityWindows(scaled, times, 0, train_i, context=context, horizon=horizon, city_id=series.city_id),
        val=CityWindows(scaled, times, train_i, val_i, context=context, horizon=horizon, city_id=series.city_id),
        context=context,
        horizon=horizon,
        unit=series.unit,
        source=series.source,
    )


def prepare_city_fold_with_scaler(
    series: CitySeries, train_end: int | str | pd.Timestamp,
    val_end: int | str | pd.Timestamp, scaler: CityScaler, *,
    context: int = CONTEXT, horizon: int = HORIZON,
    node_scale: np.ndarray | None = None,
) -> CityFold:
    """Prepare a fold using a pre-fitted scaler from an allowed data scope."""
    train_i = _boundary(series.times, train_end, "train_end")
    val_i = _boundary(series.times, val_end, "val_end")
    if val_i <= train_i:
        raise ValueError("val_end must be after train_end")
    scaled = scaler.transform(series.values[:val_i])
    fold_scaler = CityScaler(
        mean=scaler.mean.copy(), std=scaler.std.copy(), fit_end_exclusive=train_i,
        node_scale=(None if node_scale is None else np.asarray(node_scale, dtype=np.float64).copy()),
    )
    return CityFold(
        city_id=series.city_id,
        node_ids=series.node_ids,
        scaler=fold_scaler,
        train=CityWindows(scaled, series.times[:val_i], 0, train_i,
                          context=context, horizon=horizon, city_id=series.city_id),
        val=CityWindows(scaled, series.times[:val_i], train_i, val_i,
                        context=context, horizon=horizon, city_id=series.city_id),
        context=context,
        horizon=horizon,
        unit=series.unit,
        source=series.source,
    )


@dataclass(frozen=True)
class LeaveOneCityOut:
    """A pooled source tuple and one held-out target city.

    ``target.train`` always provides target-history context windows; in the
    legacy ``target_prefix_per_city`` mode it also fits target normalization,
    while ``source_only_global`` uses it only as context.  It is never part of
    ``source_folds``.  A runner should train on ``pooled_train`` and evaluate
    on ``target.val``.
    """

    source_folds: tuple[CityFold, ...]
    target: CityFold
    context: int
    horizon: int
    scaler_mode: str = "target_prefix_per_city"
    source_scaler_city_ids: tuple[str, ...] = ()

    @property
    def pooled_train(self) -> tuple[CityWindows, ...]:
        return tuple(fold.train for fold in self.source_folds)

    @property
    def source_city_ids(self) -> tuple[str, ...]:
        return tuple(fold.city_id for fold in self.source_folds)

    @property
    def provenance(self) -> dict[str, object]:
        """JSON-serializable split record for run registration.

        The record deliberately contains no target values.  It records only
        city identities, node counts, units/sources, and scaler boundaries so
        a runner can audit a pooled run without reopening any payload.
        """
        return {
            "protocol": "leave_one_city_out",
            "context": self.context,
            "horizon": self.horizon,
            "scaler_mode": self.scaler_mode,
            "node_scale_mode": ("charger_num" if self.scaler_mode == "source_only_global_capacity_normalized" else None),
            "target_history_used_for_context": True,
            "target_prefix_used_for_scaling": not self.scaler_mode.startswith("source_only_global"),
            "source_scaler_city_ids": list(self.source_scaler_city_ids),
            "source_cities": [
                {
                    "city_id": fold.city_id,
                    "node_count": len(fold.node_ids),
                    "unit": fold.unit,
                    "source": fold.source,
                    "scaler_fit_end_exclusive": fold.scaler.fit_end_exclusive,
                }
                for fold in self.source_folds
            ],
            "target_city": {
                "city_id": self.target.city_id,
                "node_count": len(self.target.node_ids),
                "unit": self.target.unit,
                "source": self.target.source,
                "scaler_fit_end_exclusive": self.target.scaler.fit_end_exclusive,
                "validation_label_start": self.target.val.label_start,
                "validation_label_end": self.target.val.label_end,
            },
        }


def iter_pooled_batches(
    protocol: LeaveOneCityOut, *, batch_size: int = 32, shuffle: bool = False,
) -> Iterator[tuple[str, CityScaler, torch.Tensor, torch.Tensor]]:
    """Yield source-city batches without padding or cross-city concatenation.

    Each yielded ``(city_id, scaler, x, y)`` keeps its city's native node
    dimension.  A model can share parameters across iterations and use the
    city id/descriptor for conditioning, while inverse scaling remains tied
    to the originating city.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for fold in protocol.source_folds:
        loader = DataLoader(fold.train, batch_size=batch_size, shuffle=shuffle)
        for x, y in loader:
            if x.device.type != "cpu" or y.device.type != "cpu":
                raise AssertionError("pooled CPU iterator unexpectedly moved tensors")
            yield fold.city_id, fold.scaler, x, y


def pooled_cpu_smoke(protocol: LeaveOneCityOut, *, batch_size: int = 4) -> list[dict[str, object]]:
    """Read one CPU batch per source city and return shape/provenance rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows: list[dict[str, object]] = []
    for fold in protocol.source_folds:
        loader = DataLoader(fold.train, batch_size=batch_size, shuffle=False)
        x, y = next(iter(loader))
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise AssertionError("pooled CPU smoke unexpectedly moved tensors")
        rows.append({
            "city_id": fold.city_id,
            "node_count": len(fold.node_ids),
            "batch_x_shape": list(x.shape),
            "batch_y_shape": list(y.shape),
            "scaler_fit_end_exclusive": fold.scaler.fit_end_exclusive,
            "source": fold.source,
        })
    return rows


def leave_one_city_out(
    cities: Mapping[str, CitySeries], target_city: str,
    train_end: BoundarySpec,
    val_end: BoundarySpec,
    *, context: int = CONTEXT, horizon: int = HORIZON,
) -> LeaveOneCityOut:
    """Build a pooled/LOCO split while preserving variable node counts.

    The function only operates on the supplied in-memory ``cities`` mapping;
    it never enumerates a data directory or infers a target city from files.
    """
    if not cities or target_city not in cities:
        raise ValueError("target_city must name one supplied city")
    if any(key != series.city_id for key, series in cities.items()):
        raise ValueError("mapping keys must match CitySeries.city_id")
    def per_city(bound: BoundarySpec, city_id: str) -> Boundary:
        if isinstance(bound, Mapping):
            if city_id not in bound:
                raise ValueError(f"missing boundary for city {city_id}")
            return bound[city_id]
        return bound

    target = prepare_city_fold(
        cities[target_city], per_city(train_end, target_city), per_city(val_end, target_city),
        context=context, horizon=horizon,
    )
    source = tuple(
        prepare_city_fold(
            cities[city_id], per_city(train_end, city_id), per_city(val_end, city_id),
            context=context, horizon=horizon,
        )
        for city_id in cities if city_id != target_city
    )
    if not source:
        raise ValueError("at least one source city is required")
    protocol = LeaveOneCityOut(
        source_folds=source, target=target, context=context, horizon=horizon,
        scaler_mode="target_prefix_per_city",
        source_scaler_city_ids=(),
    )
    # Fail closed before a caller can hand an invalid split to a trainer.
    assert_loco_isolation(protocol)
    return protocol


def leave_one_city_out_source_only(
    cities: Mapping[str, CitySeries], target_city: str,
    train_end: BoundarySpec, val_end: BoundarySpec, *,
    context: int = CONTEXT, horizon: int = HORIZON,
    node_scales: Mapping[str, np.ndarray] | None = None,
) -> LeaveOneCityOut:
    """LOCO split with a global scaler fitted only on source prefixes.

    Target history remains available as model context in its train/validation
    windows, but target values never affect the shared normalization.  This is
    the strict source-only counterpart to :func:`leave_one_city_out`.
    """
    if not cities or target_city not in cities:
        raise ValueError("target_city must name one supplied city")
    if any(key != series.city_id for key, series in cities.items()):
        raise ValueError("mapping keys must match CitySeries.city_id")
    source_ids = tuple(city_id for city_id in cities if city_id != target_city)
    if not source_ids:
        raise ValueError("at least one source city is required")
    scaler = fit_source_global_scaler(cities, source_ids, train_end, node_scales=node_scales)

    if node_scales is not None:
        missing = [city_id for city_id in cities if city_id not in node_scales]
        if missing:
            raise ValueError(f"node_scales missing cities: {missing}")
        for city_id, series in cities.items():
            values = np.asarray(node_scales[city_id]).reshape(-1)
            if values.shape != (len(series.node_ids),) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"node_scales[{city_id}] must be positive with one value per node")

    def per_city(bound: BoundarySpec, city_id: str) -> Boundary:
        if isinstance(bound, Mapping):
            if city_id not in bound:
                raise ValueError(f"missing boundary for city {city_id}")
            return bound[city_id]
        return bound

    target = prepare_city_fold_with_scaler(
        cities[target_city], per_city(train_end, target_city),
        per_city(val_end, target_city), scaler, context=context, horizon=horizon,
        node_scale=(node_scales.get(target_city) if node_scales is not None else None),
    )
    source = tuple(
        prepare_city_fold_with_scaler(
            cities[city_id], per_city(train_end, city_id),
            per_city(val_end, city_id), scaler, context=context, horizon=horizon,
            node_scale=(node_scales.get(city_id) if node_scales is not None else None),
        )
        for city_id in source_ids
    )
    protocol = LeaveOneCityOut(
        source_folds=source, target=target, context=context, horizon=horizon,
        scaler_mode=("source_only_global_capacity_normalized" if node_scales is not None else "source_only_global"),
        source_scaler_city_ids=source_ids,
    )
    assert_loco_isolation(protocol)
    return protocol


def leave_one_city_out_blended_scaler(
    cities: Mapping[str, CitySeries], target_city: str,
    train_end: BoundarySpec, val_end: BoundarySpec, *, target_weight: float,
    context: int = CONTEXT, horizon: int = HORIZON,
) -> LeaveOneCityOut:
    """LOCO with source-global source folds and blended target scaling.

    The source representation remains shared/global.  Only the held-out
    target fold uses a convex blend of source-global and target-prefix
    mean/std, preventing the target scaler from silently re-normalizing every
    source city independently.
    """
    if not 0.0 <= float(target_weight) <= 1.0:
        raise ValueError("target_weight must be in [0,1]")
    source_protocol = leave_one_city_out_source_only(
        cities, target_city, train_end, val_end, context=context, horizon=horizon,
    )
    target_city_scaler = fit_city_scaler(
        cities[target_city].values,
        _boundary(cities[target_city].times, train_end[target_city] if isinstance(train_end, Mapping) else train_end, "train_end"),
    )
    source_scaler = source_protocol.target.scaler
    weight = float(target_weight)
    mean = (1.0 - weight) * source_scaler.mean + weight * target_city_scaler.mean
    std = (1.0 - weight) * source_scaler.std + weight * target_city_scaler.std
    blended = CityScaler(
        mean=np.asarray(mean, dtype=np.float64),
        std=np.maximum(np.asarray(std, dtype=np.float64), 1e-6),
        fit_end_exclusive=target_city_scaler.fit_end_exclusive,
    )
    def per_city(bound: BoundarySpec, city_id: str) -> Boundary:
        if isinstance(bound, Mapping):
            if city_id not in bound:
                raise ValueError(f"missing boundary for city {city_id}")
            return bound[city_id]
        return bound
    target = prepare_city_fold_with_scaler(
        cities[target_city], per_city(train_end, target_city),
        per_city(val_end, target_city), blended, context=context, horizon=horizon,
    )
    protocol = LeaveOneCityOut(
        source_folds=source_protocol.source_folds,
        target=target, context=context, horizon=horizon,
        scaler_mode=f"source_target_blend_{weight:.3f}",
        source_scaler_city_ids=source_protocol.source_city_ids,
    )
    assert_loco_isolation(protocol)
    return protocol


def assert_loco_isolation(protocol: LeaveOneCityOut) -> None:
    """Raise if a prepared split accidentally includes its target as source.

    This is intentionally a cheap protocol guard that can run before a
    trainer is launched.  It also checks that all source/target window
    contracts agree; node counts are allowed to differ by city.
    """
    if protocol.target.city_id in protocol.source_city_ids:
        raise AssertionError("held-out target city leaked into pooled sources")
    if not protocol.source_folds:
        raise AssertionError("LOCO protocol has no source cities")
    for fold in protocol.source_folds + (protocol.target,):
        if fold.context != protocol.context or fold.horizon != protocol.horizon:
            raise AssertionError("city folds disagree on context/horizon")
        if fold.scaler.fit_end_exclusive > fold.val.label_start:
            raise AssertionError("scaler extends beyond the training prefix")
        if fold.train.label_end != fold.val.label_start:
            raise AssertionError("train/validation label boundary is not contiguous")


def city_series_from_loa(loa_series: object, city_id: str = "LOA") -> CitySeries:
    """Adapt an already-loaded LOA ``LOASeries`` without opening its ZIP."""
    try:
        times = loa_series.times
        node_ids = tuple(loa_series.site_ids)
        values = loa_series.values
    except AttributeError as error:
        raise TypeError("loa_series must expose times, site_ids, and values") from error
    return CitySeries(city_id, times, node_ids, values, source="CHARGED:LOA/volume.csv")


__all__ = [
    "CONTEXT", "HORIZON", "CitySeries", "CityScaler", "CityWindows", "CityFold",
    "LeaveOneCityOut", "fit_city_scaler", "fit_source_global_scaler",
    "prepare_city_fold", "prepare_city_fold_with_scaler", "leave_one_city_out",
    "leave_one_city_out_source_only",
    "assert_loco_isolation", "iter_pooled_batches", "pooled_cpu_smoke",
    "city_series_from_loa",
]

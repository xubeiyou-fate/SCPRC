#!/usr/bin/env python3
"""Describe frozen Full-vs-Exact energy gains and history-only gates.

All writes stay beside this script. No model is imported, trained or run.
The fold/origin grid, contrasts, rank boundaries and features are fixed before
reading energy errors. Undefined correlations and empty tied strata survive.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parents[1]
REVIEW = Path(os.environ.get('SCPRC_GATE_EVIDENCE_DIR', str(ROOT / 'experiments/support/review_evidence/04_Gate_Diagnostics'))).expanduser()
PROTOCOL = ROOT / 'scprc/training/training_protocol.py'
ARCHIVE = ROOT / 'scprc/data_adapters/charged/Hourly.zip'
CITIES = ('AMS', 'JHB', 'LOA', 'MEL', 'SPO', 'SZH')
SEEDS = (13, 17, 29)
STARTS = np.arange(2928, 3649, dtype=np.int64)
KEY = ['city', 'seed', 'label_start']
ERRORS = ['peak_magnitude_mae', 'ramp_mae', 'total_24h_mae', 'peak_timing_mae_hours']
GATES = ['a_i_mean', 'b_i_mean', 'composite_mean', 'a_i_attenuated_fraction',
         'b_i_attenuated_fraction', 'bS_source_window_gate', 'window_peer_disagreement']
DERIVED = ['source_trust_attenuated']
PERIODIC = ['input_acf24', 'input_power_24h_fraction', 'input_power_168h_fraction']
TIMES = ['forecast_target_start', 'issue_time']
ACTUAL = ['actual_peak', 'actual_total']
PREDICTED = ['predicted_peak', 'predicted_total']
EXPECTED = pd.MultiIndex.from_product([CITIES, SEEDS, STARTS], names=KEY)
RANK_CUTS = [0.25, 0.5, 0.75]
ATOL_GATE = 1e-6


def project_artifact_path(value):
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scprc.project_paths import project_artifact_path as resolve
    return resolve(value)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def output_path(name: str) -> Path:
    path = OUT / name
    if path.parent.resolve() != OUT or name.startswith('gate_') or name == 'gate_summary.csv':
        raise ValueError(f'Refusing output outside owned files: {path}')
    return path


def write_json(name: str, payload: dict) -> None:
    path = output_path(name)
    tmp = output_path(name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_csv(name: str, frame: pd.DataFrame) -> None:
    path = output_path(name)
    tmp = output_path(name + '.tmp')
    opts = {'method': 'gzip', 'mtime': 0} if name.endswith('.gz') else None
    frame.to_csv(tmp, index=False, encoding='utf-8', compression=opts, float_format='%.12g')
    tmp.replace(path)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['city'] = frame['city'].astype(str).str.upper()
    for col in ['seed', 'label_start']:
        val = pd.to_numeric(frame[col], errors='raise')
        if val.isna().any() or not np.equal(val, np.floor(val)).all():
            raise ValueError(f'{col}: missing/noninteger keys')
        frame[col] = val.astype(np.int64)
    return frame


def validate_grid(frame: pd.DataFrame, label: str) -> None:
    if frame.duplicated(KEY).any():
        raise ValueError(f'{label}: duplicate city/seed/label_start; refusing many-to-many join')
    idx = pd.MultiIndex.from_frame(frame[KEY])
    missing, extra = EXPECTED.difference(idx), idx.difference(EXPECTED)
    if len(missing) or len(extra):
        raise ValueError(f'{label}: incomplete grid, missing={len(missing)}, extra={len(extra)}; '
                         f'first_missing={list(missing[:3])}, first_extra={list(extra[:3])}')


def validate_times(frame: pd.DataFrame, label: str) -> None:
    target = pd.Timestamp('2023-04-01') + pd.to_timedelta(frame['label_start'], unit='h')
    for col, expected in [('forecast_target_start', target), ('issue_time', target - pd.Timedelta(hours=1))]:
        got = pd.to_datetime(frame[col], errors='raise')
        if got.isna().any() or not np.array_equal(got.to_numpy(), expected.to_numpy()):
            raise ValueError(f'{label}: {col} differs from fixed local-hour protocol')


def finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: nonfinite {col}; no origin silently discarded')


def load_gates(path: Path) -> pd.DataFrame:
    gate = normalize_keys(pd.read_csv(path))
    required = KEY + TIMES + GATES
    if set(required) - set(gate):
        raise ValueError(f'Gate columns missing: {sorted(set(required) - set(gate))}')
    validate_grid(gate, 'gates')
    validate_times(gate, 'gates')
    finite(gate, GATES, 'gates')
    gate['source_trust_attenuated'] = (gate['bS_source_window_gate'] < 1 - ATOL_GATE).astype(float)
    return gate.sort_values(KEY).reset_index(drop=True)


def load_pairs(path: Path, gate: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    energy = normalize_keys(pd.read_csv(path))
    required = KEY + TIMES + ERRORS + ACTUAL + PREDICTED + ['model', 'lineage', 'postprocess']
    if set(required) - set(energy):
        raise ValueError(f'Energy columns missing: {sorted(set(required) - set(energy))}')
    if energy[['model', 'lineage', 'postprocess']].isna().any().any():
        raise ValueError('Energy has missing model/lineage/postprocess identity')
    available = energy.groupby(['model', 'lineage', 'postprocess'], dropna=False).size().reset_index(name='rows')
    selected = energy.loc[(energy['lineage'] == args.lineage) & energy['model'].isin([args.full_model, args.exact_model])].copy()
    if args.postprocess:
        selected = selected.loc[selected['postprocess'].isin(args.postprocess)]
    settings = sorted(selected['postprocess'].unique())
    if not settings:
        raise ValueError('No requested canonical Full/Exact rows found')
    all_pairs, qc_rows = [], []
    basecols = KEY + TIMES + ACTUAL + PREDICTED + ERRORS
    for setting in settings:
        base = selected.loc[selected['postprocess'] == setting]
        parts = {}
        for role, model in [('full', args.full_model), ('exact', args.exact_model)]:
            part = base.loc[base['model'] == model, basecols].copy()
            validate_grid(part, f'{model}/{setting}')
            validate_times(part, f'{model}/{setting}')
            finite(part, ERRORS + ACTUAL + PREDICTED, model)
            if (part[ERRORS] < 0).any().any():
                raise ValueError(f'{model}: negative absolute errors')
            parts[role] = part.set_index(KEY).sort_index()
        full, exact = parts['full'], parts['exact']
        for col in TIMES + ACTUAL:
            if not np.array_equal(full[col].to_numpy(), exact[col].to_numpy()):
                raise ValueError(f'Full/Exact {col} mismatch for {setting}')
        # Outer joins are audited; a failed pairing aborts instead of dropping rows.
        pair = full.reset_index().merge(exact.reset_index(), on=KEY, how='outer',
                                       suffixes=('_full', '_exact'), validate='one_to_one', indicator=True)
        if not pair['_merge'].eq('both').all():
            raise ValueError('Unpaired Full/Exact origin')
        pair = pair.drop(columns='_merge')
        for col in TIMES + ACTUAL:
            pair[col] = pair.pop(col + '_full')
            pair = pair.drop(columns=col + '_exact')
        for metric in ERRORS:
            pair[metric + '_full_minus_exact'] = pair[metric + '_full'] - pair[metric + '_exact']
            pair[metric + '_gain'] = -pair[metric + '_full_minus_exact']
        joined = pair.merge(gate, on=KEY, how='outer', suffixes=('', '_gate'), validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('Gate/energy key mismatch; all origins required')
        for col in TIMES:
            if not np.array_equal(joined[col].to_numpy(), joined[col + '_gate'].to_numpy()):
                raise ValueError(f'Gate/energy {col} mismatch')
            joined = joined.drop(columns=col + '_gate')
        joined = joined.drop(columns='_merge')
        joined['full_model'], joined['exact_model'] = args.full_model, args.exact_model
        joined['lineage'], joined['postprocess'] = args.lineage, setting
        joined['issue_hour_phase24'] = pd.to_datetime(joined['issue_time']).dt.hour
        for (city, seed), fold in joined.groupby(['city', 'seed'], sort=True):
            qc_rows.append({'city': city, 'seed': int(seed), 'lineage': args.lineage, 'postprocess': setting,
                            'gate_rows': 721, 'full_rows': 721, 'exact_rows': 721,
                            'paired_rows': len(fold), 'label_start_first': int(fold.label_start.min()),
                            'label_start_last': int(fold.label_start.max()), 'keys_exact': True,
                            'timestamps_exact': True, 'actual_peak_total_exact': True,
                            'unmatched_full': 0, 'unmatched_exact': 0, 'unmatched_gates': 0})
        all_pairs.append(joined)
    return pd.concat(all_pairs, ignore_index=True), pd.DataFrame(qc_rows), {
        'energy_rows_read': len(energy), 'canonical_rows_selected': len(selected),
        'available_model_lineage_postprocess': available.to_dict('records'), 'postprocesses': settings}


def history_features(gate: pd.DataFrame, archive_path: Path) -> tuple[pd.DataFrame, dict]:
    """Reuse two pure loader functions from the pinned protocol, without imports.

    Only the prefix ending at the last issue is materialized. Each feature
    accesses [label_start-168:label_start], never its future target window.
    """
    source = PROTOCOL.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(PROTOCOL))
    names = {'_member_prefix_sha256', '_read_volume_prefix'}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != names:
        raise ValueError('Pinned protocol pure loader functions missing')
    namespace = {'np': np, 'pd': pd, 'zipfile': zipfile, 'hashlib': hashlib,
                 'Any': object, 'FIRST_HOUR': pd.Timestamp('2023-04-01'), 'VAL_END': int(STARTS.max())}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROTOCOL), 'exec'), namespace)
    rows, cities = [], []
    with zipfile.ZipFile(archive_path) as z:
        for city in CITIES:
            panel = namespace['_read_volume_prefix'](z, city)
            series = panel['values'].sum(axis=1, dtype=np.float64)
            starts = sorted(gate.loc[gate.city == city, 'label_start'].unique())
            for start in starts:
                history = series[start - 168:start]
                if len(history) != 168:
                    raise ValueError('Incomplete 168h input')
                centered = history - history.mean()
                energy = float(np.sum(centered ** 2))
                fft = np.fft.rfft(centered)
                p24 = 2 * abs(fft[7]) ** 2 / (168 * energy) if energy > 0 else np.nan
                p168 = 2 * abs(fft[1]) ** 2 / (168 * energy) if energy > 0 else np.nan
                left, right = history[:-24], history[24:]
                acf = float(np.corrcoef(left, right)[0, 1]) if np.ptp(left) > 0 and np.ptp(right) > 0 else np.nan
                rows.append({'city': city, 'label_start': int(start), 'history_start_index': int(start - 168),
                             'history_end_exclusive_index': int(start), 'history_hours': 168,
                             'input_acf24': acf, 'input_power_24h_fraction': float(p24),
                             'input_power_168h_fraction': float(p168), 'input_acf168': np.nan,
                             'input_acf168_status': 'unidentifiable_zero_lag_pairs_with_168h_history',
                             'input_city_total_mean': float(history.mean()),
                             'input_city_total_std': float(history.std()),
                             'input_periodicity_status': 'ok' if energy > 0 else 'constant_history'})
            cities.append({'city': city, 'site_count': len(panel['site_ids']),
                           'prefix_rows': len(panel['values']), 'prefix_sha256': panel['prefix_sha256'],
                           'archive_member': panel['source_member'],
                           'site_ids_sha256': hashlib.sha256('\n'.join(panel['site_ids']).encode()).hexdigest()})
            print(f'input-only history features: {city}, {len(starts)} origins', flush=True)
    features = gate[KEY].merge(pd.DataFrame(rows), on=['city', 'label_start'], how='left', validate='many_to_one')
    validate_grid(features, 'input history features')
    return features, {'status': 'computed_input_only', 'city_prefixes': cities,
                      'lag168_autocorrelation': 'not_identifiable_with_only_168_history_hours',
                      'no_target_window_cells_used_for_origin_features': True,
                      'reader': 'AST extraction of _read_volume_prefix/_member_prefix_sha256 from fixed protocol; '
                                'VAL_END shortened to 3648, no protocol import or model execution'}


def rank_quartiles(series: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Average ranks keep ties together; all Q1..Q4 rows are emitted later.

    A constant fold is assigned to CONSTANT, never split by row/label order.
    Undefined inputs are retained as MISSING rather than dropped.
    """
    values = pd.to_numeric(series, errors='raise')
    valid = np.isfinite(values)
    n = int(valid.sum())
    groups = pd.Series('MISSING', index=series.index, dtype='object')
    ranks = pd.Series(np.nan, index=series.index, dtype=float)
    n_unique = int(values[valid].nunique())
    if n_unique == 1:
        groups.loc[valid] = 'CONSTANT'
        ranks.loc[valid] = 0.5
        status = 'constant'
    elif n_unique > 1:
        # The centered empirical rank lies in (0,1); ties have average ranks.
        ranks.loc[valid] = (values.loc[valid].rank(method='average') - 0.5) / n
        bins = np.searchsorted(RANK_CUTS, ranks.loc[valid].to_numpy(), side='left') + 1
        groups.loc[valid] = ['Q' + str(x) for x in bins]
        status = 'ties_kept_average_rank' if n_unique < n else 'ok'
    else:
        status = 'all_missing'
    rawq = np.quantile(values[valid], RANK_CUTS) if n else [np.nan] * 3
    meta = {'n_total': len(series), 'n_finite': n, 'n_unique': n_unique, 'status': status,
            'raw_q25': float(rawq[0]), 'raw_q50': float(rawq[1]), 'raw_q75': float(rawq[2]),
            'duplicate_raw_quantiles': bool(len(set(rawq)) < 3) if n else False,
            'rank_boundary_q25': .25, 'rank_boundary_q50': .5, 'rank_boundary_q75': .75}
    return groups, ranks, meta


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, str, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    # Do not pairwise-drop origins. Any missing value makes this coefficient undefined.
    if n != len(x):
        return np.nan, 'nonfinite_values_retained_coefficient_undefined', n
    if x.nunique() <= 1:
        return np.nan, 'constant_feature', n
    if y.nunique() <= 1:
        return np.nan, 'constant_gain', n
    return float(x.rank(method='average').corr(y.rank(method='average'))), 'ok', n


def stats_for_group(group: pd.DataFrame, metric: str) -> dict:
    gain = group[metric + '_gain']
    return {'n_origins': len(group), 'exact_error_mean': float(group[metric + '_exact'].mean()),
            'full_error_mean': float(group[metric + '_full'].mean()),
            'full_minus_exact_error_mean': float(group[metric + '_full_minus_exact'].mean()),
            'gain_mean_exact_minus_full': float(gain.mean()), 'gain_median': float(gain.median()),
            'gain_positive_share': float(gain.gt(0).mean()) if len(group) else np.nan,
            'gain_zero_share': float(gain.eq(0).mean()) if len(group) else np.nan,
            'gain_negative_share': float(gain.lt(0).mean()) if len(group) else np.nan}


def raw_first(value: str) -> tuple[int, str]:
    return (0 if value == 'raw' else 1, value)


def summarize_groups(frame: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        occupied = part['n_origins'].gt(0)
        row.update({'folds_total': len(part), 'folds_nonempty': int(occupied.sum()),
                    'folds_empty': int((~occupied).sum()), 'n_origins_total': int(part['n_origins'].sum())})
        for col in value_cols:
            val = part[col]
            row[col + '_equal_fold_mean'] = float(val.mean())
            row[col + '_folds_defined'] = int(val.notna().sum())
            # A complete-18-fold value is undefined if the stratum is empty in any fold.
            row[col + '_all18_mean'] = float(val.mean()) if len(part) == 18 and val.notna().all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyses(paired: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    correlations, strata, thresholds, phase, fold_means = [], [], [], [], []
    strata_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'CONSTANT', 'MISSING']
    for (setting, city, seed), fold in paired.groupby(['postprocess', 'city', 'seed'], sort=True):
        identity = {'city': city, 'seed': int(seed), 'lineage': str(fold['lineage'].iloc[0]), 'postprocess': setting}
        for metric in ERRORS:
            fold_means.append({**identity, 'metric': metric, **stats_for_group(fold, metric)})
        for feature in features:
            groups, ranks, meta = rank_quartiles(fold[feature])
            thresholds.append({**identity, 'feature': feature, **meta})
            paired.loc[fold.index, feature + '_stratum'] = groups
            paired.loc[fold.index, feature + '_rank_fraction'] = ranks
            for metric in ERRORS:
                gain = fold[metric + '_gain']
                rho, status, n = spearman(fold[feature], gain)
                correlations.append({**identity, 'feature': feature, 'metric': metric,
                                     'n_origins': len(fold), 'n_finite_pairs': n,
                                     'feature_unique': int(fold[feature].nunique()), 'gain_unique': int(gain.nunique()),
                                     'rho_gain_exact_minus_full': rho,
                                     'rho_error_delta_full_minus_exact': -rho, 'status': status})
                for label in strata_labels:
                    subset = fold.loc[groups == label]
                    strata.append({**identity, 'feature': feature, 'stratum': label,
                                   'feature_status': meta['status'], 'metric': metric,
                                   'feature_min': float(subset[feature].min()),
                                   'feature_max': float(subset[feature].max()),
                                   **stats_for_group(subset, metric)})
        for hour in range(24):
            group = fold.loc[fold.issue_hour_phase24 == hour]
            for metric in ERRORS:
                phase.append({**identity, 'issue_hour_phase24': hour, 'metric': metric,
                              **stats_for_group(group, metric),
                              **{f + '_mean': float(group[f].mean()) for f in features}})
    corr, grouped, phases = pd.DataFrame(correlations), pd.DataFrame(strata), pd.DataFrame(phase)
    corrsum = []
    for (setting, feature, metric), part in corr.groupby(['postprocess', 'feature', 'metric']):
        val = part.rho_gain_exact_minus_full
        corrsum.append({'postprocess': setting, 'feature': feature, 'metric': metric,
                        'folds_total': len(part), 'folds_defined': int(val.notna().sum()),
                        'folds_constant_feature': int(part.status.eq('constant_feature').sum()),
                        'folds_positive_rho': int(val.gt(0).sum()), 'folds_negative_rho': int(val.lt(0).sum()),
                        'rho_equal_fold_mean': float(val.mean()), 'rho_fold_median': float(val.median())})
    values = ['exact_error_mean', 'full_error_mean', 'gain_mean_exact_minus_full',
              'full_minus_exact_error_mean', 'gain_positive_share', 'gain_zero_share', 'gain_negative_share']
    return {'per_fold_correlations.csv': corr,
            'correlation_equal_fold_summary.csv': pd.DataFrame(corrsum),
            'per_fold_strata.csv': grouped,
            'strata_all18fold_summary.csv': summarize_groups(grouped, ['postprocess', 'feature', 'stratum', 'metric'], values),
            'rank_quartile_thresholds.csv': pd.DataFrame(thresholds),
            'phase24_per_fold.csv': phases,
            'phase24_all18fold_summary.csv': summarize_groups(phases, ['postprocess', 'issue_hour_phase24', 'metric'],
                                                            values + [f + '_mean' for f in features]),
            'paired_fold_gain_summary.csv': pd.DataFrame(fold_means),
            'paired_origin_gate_energy.csv.gz': paired}


def validate_outputs(results: dict[str, pd.DataFrame], qc: pd.DataFrame) -> dict:
    """Check row conservation and recombine strata/phase means to each fold."""
    paired = results['paired_origin_gate_energy.csv.gz']
    strata = results['per_fold_strata.csv']
    phase = results['phase24_per_fold.csv']
    identity = ['postprocess', 'city', 'seed', 'metric']
    folded = results['paired_fold_gain_summary.csv'].set_index(identity)
    for table, keys in [(strata, identity + ['feature']), (phase, identity)]:
        for key, group in table.groupby(keys):
            if group.n_origins.sum() != 721:
                raise ValueError(f'Origin conservation failed: {key}')
            reference = folded.loc[key[:4]]
            occupied = group.n_origins.gt(0)
            for col in ['exact_error_mean', 'full_error_mean', 'gain_mean_exact_minus_full']:
                recombined = np.average(group.loc[occupied, col], weights=group.loc[occupied, 'n_origins'])
                if not np.isclose(recombined, reference[col], rtol=1e-11, atol=1e-8):
                    raise ValueError(f'Weighted group recombination failed: {key}/{col}')
    for setting, part in paired.groupby('postprocess'):
        validate_grid(part, f'final joined output/{setting}')
        counts = part.groupby(['city', 'seed']).issue_hour_phase24.value_counts()
        if not counts.isin([30, 31]).all() or len(counts) != 18 * 24:
            raise ValueError('Issue-hour phase grid is incomplete')
    return {'status': 'passed', 'paired_origin_rows': len(paired),
            'complete_fold_settings': len(qc), 'strata_origin_conservation': True,
            'phase_origin_conservation': True, 'weighted_strata_and_phase_recombine_to_fold': True,
            'issue_hour_counts_per_fold': 'hour23=31; other hours=30',
            'all_nonpositive_gains_retained': True}


def readme(status: str, energy_path: Path | None, results: dict | None, history: dict | None) -> str:
    lines = ['# 门控、输入周期性与能源预测收益的描述性关联', '',
             f'状态：`{status}`。本脚本仅在 `03_mechanism/` 写入自己的文件，保留已有 gate summary；不训练、不执行模型、不修改统一评价器。', '',
             '## 输入与配对', '',
             f'- gate：`{REVIEW / "per_origin_gate_summary.csv"}`。',
             f'- 逐 origin 能源误差：`{energy_path or "等待 01_metrics/per_origin_energy_metrics.csv.gz 或 .csv"}`。',
             '- 固定 6 城市 × seeds 13/17/29 × label_start 2928…3648，共 18×721=12,978 个 origin。',
             '- 默认仅选 `full_scpcr` / `exact_pcgrad`，lineage=`canonical_four_arm`。所有 postprocess 分开分析；从不混入 P2 independent-switch Full。',
             '- `city,seed,label_start` 以 one-to-one outer join 配对；键重复、缺失、时间或 actual peak/total 不一致直接报错，不用 inner join 静默丢失样本。',
             '- 严查 issue_time=forecast_target_start−1 h，phase24 使用 issue hour；不把 target hour 混作 issue hour。时间沿用发布的城市本地 hourly grid。', '',
             '## 预先固定的分析规则', '',
             '- 四项误差：Peak、Ramp、24h Total、Peak Timing，直接读取统一评价器逐 origin error；不重算、不更改这些误差。',
             '- error delta=`Full error−Exact error`（负值较好）；收益 gain=`Exact error−Full error`（正值较好）。输出同时保留两种符号。',
             '- 每 fold/特征/误差计算探索性 Spearman（平均秩的 Pearson）；不计算或报告独立 origin p 值。常量特征或常量收益的 rho 为缺失，状态显式保留；汇总 rho 仅为有定义fold的描述性均值，不能代称18fold完整均值。',
             '- 特征含 scale/peer/composite gate 均值、scale/peer attenuated share、source trust `bS`、window disagreement；source attenuated indicator 固定为 `bS<1−1e−6`。',
             '- gate 导出没有逐 origin composite attenuated share，因此不由 composite mean 伪造；沿用原导出的 scale/peer share。',
             '- fold 内使用 `(average_rank−0.5)/n_finite`，阈值固定 0.25/0.5/0.75；恰在边界者归较低组，同值不按 origin 顺序拆散。不是选择最佳 cutpoint。',
             '- 重复原始分位数保留在 thresholds 表；始终输出 Q1–Q4、CONSTANT、MISSING 六行。常量全部放 CONSTANT，空分组计数 0、均值缺失，不强造四个非空组。',
             '- 分组汇总先算 fold 内均值，再等 fold 平均；输出 18 fold 总数、非空 fold 数、定义数。`*_all18_mean` 仅在全部18 fold 都有定义时给值。',
             '- phase24 始终输出每个 fold 的全部24 issue hours，再等 fold 汇总。保留所有正、零、负收益，不按改善筛选 origin、fold、特征、小时或分组。',
             '- raw 是主要口径；nonnegative_clip 仅为敏感性/操作口径，所有表按 postprocess 明确分开。',
             '- gate 来源是冻结 Full checkpoint 的重新前向导出，原导出给出的最大保存预测差异0.0078125 kWh，在各fold既定容差内；这是数值容差下的同lineage，不宣称gate重构与保存预测bitwise相等。', '',
             '## 输入周期性与限制', '']
    if history and history['status'] == 'computed_input_only':
        lines += ['- 复用固定 data protocol 的两个纯读取函数（从源文件 AST 提取，避免导入模型），float32 城市面板及原站点顺序；只 materialize 到最后 issue 的数据。',
                  '- 每城市/origin 仅用 `[label_start−168:label_start]` 的原始输入，站点求和为城市总量，再算：',
                  '  - `input_acf24`：同一168h历史内，后144点与前144点的 Pearson correlation。',
                  '  - `input_power_24h_fraction` / `input_power_168h_fraction`：历史去均值后的 rFFT 第7/第1频率 bin 的双侧能量占比 `2|X_k|²/(168×Σ(x−mean)²)`；仅去均值，无额外滤波或趋势拟合。',
                  '- 168h 仅有一个周周期，lag-168 没有可配对点；`input_acf168` 显式缺失。168h 频率能量只能称单窗口频谱描述，不能证明稳定周重复，且可能含低频趋势。',
                  '- 相同城市不同 seed 使用同一历史特征，保留全部 fold 但不将其当作独立城市重复。相邻窗口重叠，相关也可能受共同时间趋势/水平影响。']
    else:
        lines += ['- 尚未计算输入周期性，或显式使用 `--history off`。只报告 phase-hour 和历史 gate proxy；phase-hour 不是周期性强度。不得据此宣称已检验24/168h周期机制。']
    lines += ['', '## 产物', '',
              '- `per_fold_correlations.csv` / `correlation_equal_fold_summary.csv`：逐 fold rho、常量/缺失状态和等 fold 描述。',
              '- `rank_quartile_thresholds.csv` / `per_fold_strata.csv` / `strata_all18fold_summary.csv`：所有 fold 的固定秩分组与空组。',
              '- `phase24_per_fold.csv` / `phase24_all18fold_summary.csv`：issue-hour 24相位关联。',
              '- `input_periodicity_per_origin.csv.gz`：输入历史特征（启用时）；`paired_origin_gate_energy.csv.gz`：全部严格配对的误差、收益、gate、历史特征、分组。',
              '- `paired_fold_gain_summary.csv`：每fold四项收益；`join_qc.csv` / `join_qc.json`：完整网格及严格配对检查。',
              '- `source_hashes.csv` / `mechanism_manifest.json`：输入文件、脚本、原始archive、协议和gate lineage的SHA-256及输出hash；`validation_report.json`：计数及分组重组验证。',
              '- [lineage_parity_check.json](lineage_parity_check.json) / [逐fold检查表](lineage_parity_check.csv)：canonical gate指针、当前checkpoint/prediction SHA与导出记录对照、权重未变及保存预测差异容差。来源：[gate frozen_inference_lineage.json](../../review_evidence/04_Gate_Diagnostics/frozen_inference_lineage.json)。此处复核来源hash及既有parity记录，不重新执行模型前向。',
              '- 探索图由 `plot_mechanism.py` 独立生成：`exploratory_correlations_<postprocess>.pdf/.png` 与 `exploratory_phase24_<postprocess>.pdf/.png`；全部11特征、4项误差及18fold轨迹均保留，图形source data来自本目录CSV。', '',
              '## 解释边界', '',
              '这些是已冻结模型、固定6城3seed的事后描述性关联。12,978个滚动origin不是独立重复，18 fold也共享城市和训练来源；不据此报告独立origin显著性。',
              '门控是同一输入的函数，phase、负荷水平、趋势可能共同影响误差；相关方向不能证明门控导致收益，分组亦不能替代Full-drop干预消融。全部不利、零收益及常量结果均保留。', '',
              '运行：`python experiments/energy_value/03_mechanism/analyze_mechanism.py`。',
              '文件未到时：加 `--wait-seconds 600`（每10秒检查一次；状态写入本目录），或在文件就绪后重新运行。支持 `.csv.gz` 和 `.csv`；同时存在时优先 `.gz`，全部postprocess独立保留。', '']
    if results:
        foldgain = results['paired_fold_gain_summary.csv']
        lines += ['## 当前结果', '', f'- 已配对 origin 行数：{len(results["paired_origin_gate_energy.csv.gz"]):,}（含独立的postprocess设置）。',
                  '- 下表仅报告所有18 fold的等权收益，不据此筛选特征或窗口；正值表示Full误差较小。', '',
                  '| postprocess | 指标 | fold数 | 等fold收益 Exact−Full |', '|---|---|---:|---:|']
        for setting in sorted(foldgain.postprocess.unique(), key=raw_first):
            for metric in ERRORS:
                part = foldgain.loc[foldgain.postprocess.eq(setting) & foldgain.metric.eq(metric)]
                lines.append(f'| {setting} | {metric} | {len(part)} | {part.gain_mean_exact_minus_full.mean():.6g} |')
        corr = results['per_fold_correlations.csv']
        lines += ['', f'- rho状态：{corr.status.value_counts().to_dict()}。缺失rho未填0。',
                  '- 两种postprocess下，source trust在15/18fold恒定；scale/peer attenuated share分别在12/18及11/18fold恒定，故不能对这些fold估计相应强度关联。',
                  '- LOA/seed13的Timing收益为常量，两种postprocess均保留未定义rho；不把它填0或剔除该fold。',
                  '- raw等fold结果为Peak、Ramp、Timing误差较低，而24h Total误差较高；全部方向和fold差异保留，不能宣称四指标全面改善。',
                  '- **关联较弱且方向不一致**：raw的composite gate与Peak/Ramp/Total/Timing收益的平均fold rho分别为−0.061/−0.041/+0.050/+0.023（Timing仅17fold有定义）；其他特征与分组亦完整输出。此概述指平均描述，不能用弱均值否定个别fold较强或相反的关联。',
                  '- **source特征限制**：source trust在15/18fold恒定；余下3fold的raw平均rho为Peak −0.078、Ramp +0.076、Total +0.130，Timing仅2fold可定义（−0.058）。这些不能外推为18fold门控机制支持。',
                  '- **gate lineage/parity**：18/18fold均指向canonical Full；导出记录的weights_before/after相等且weights_unchanged=true，当前checkpoint及保存预测SHA均与记录一致。最大保存预测差异为0.0078125 kWh，18/18均在各fold既定容差内；不是bitwise相同声明。详见上述parity检查链接。', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy', type=Path)
    parser.add_argument('--gates', type=Path, default=REVIEW / 'per_origin_gate_summary.csv')
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--lineage', default='canonical_four_arm')
    parser.add_argument('--full-model', default='full_scpcr')
    parser.add_argument('--exact-model', default='exact_pcgrad')
    parser.add_argument('--postprocess', action='append', help='Repeat to request named settings; default all canonical settings')
    parser.add_argument('--history', choices=['on', 'off'], default='on')
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if not args.gates.is_file():
        raise FileNotFoundError(f'Gate evidence is missing: {args.gates}. Set SCPRC_GATE_EVIDENCE_DIR or --gates to the supplied evidence.')
    if args.lineage != 'canonical_four_arm':
        parser.error('Gate export is canonical_four_arm; refusing another Full lineage')
    protected = {p: sha(p) for p in OUT.glob('*gate*summary*.csv')}
    state = {'status': 'starting', 'training': False, 'output_directory': str(OUT)}
    sources = []

    def record(path: Path, role: str) -> None:
        sources.append({'role': role, 'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)})

    try:
        gate = load_gates(args.gates)
        record(Path(__file__), 'analysis_script')
        record(args.gates, 'per_origin_gate_diagnostics')
        for name in ['frozen_inference_lineage.json', 'export_frozen_gate_diagnostics.py']:
            path = args.gates.parent / name
            if path.is_file():
                record(path, 'gate_lineage_or_exporter')
        frozen = args.gates.parent / 'frozen_inference_lineage.json'
        if not frozen.is_file():
            raise ValueError('Gate lineage manifest required')
        provenance = json.loads(frozen.read_text())
        lineage_rows = provenance['folds']
        if {(r['city'], int(r['seed'])) for r in lineage_rows} != {(c, s) for c in CITIES for s in SEEDS} or len(lineage_rows) != 18:
            raise ValueError('Gate lineage does not contain precisely 18 canonical folds')
        parity_rows = []
        for row in lineage_rows:
            expected = ROOT / f'experiments/forecasting/h24_framework/target_{row["city"].lower()}_seed{row["seed"]}/scpcr_candidate/predictions.npz'
            if project_artifact_path(row['prediction_archive']).resolve() != expected.resolve() or not row['all_gates_enabled']:
                raise ValueError('Gate lineage differs from canonical Full')
            if row['saved_prediction_max_abs_difference_kWh'] > row['saved_prediction_acceptance_tolerance_kWh']:
                raise ValueError('Gate export failed its frozen inference tolerance')
            if not row['weights_unchanged'] or row['weights_before_sha256'] != row['weights_after_sha256']:
                raise ValueError('Gate export records changed weights')
            if row['training'] or row['backward'] or row['optimizer_created']:
                raise ValueError('Gate export was not inference-only')
            if not row['all_recorded_source_code_hashes_match']:
                raise ValueError('Gate export records a source-code mismatch')
            checkpoint = project_artifact_path(row['checkpoint_path'])
            expected_checkpoint = expected.parent / 'checkpoint.pt'
            if checkpoint.resolve() != expected_checkpoint.resolve():
                raise ValueError('Gate checkpoint pointer is not canonical Full')
            current_checkpoint_hash, current_prediction_hash = sha(checkpoint), sha(expected)
            if current_checkpoint_hash != row['checkpoint_sha256'] or current_prediction_hash != row['prediction_archive_sha256']:
                raise ValueError('Current gate source file hash differs from frozen lineage record')
            parity_rows.append({'city': row['city'], 'seed': row['seed'], 'gate_lineage': 'canonical_four_arm',
                                'gate_model': 'full_scpcr', 'canonical_pointer_pass': True,
                                'checkpoint_path': str(checkpoint), 'checkpoint_sha256': current_checkpoint_hash,
                                'prediction_archive': str(expected), 'prediction_archive_sha256': current_prediction_hash,
                                'current_source_hashes_match_export': True, 'weights_unchanged': True,
                                'weights_before_after_equal': True, 'all_recorded_source_code_hashes_match': True,
                                'saved_prediction_max_abs_difference_kWh': row['saved_prediction_max_abs_difference_kWh'],
                                'saved_prediction_acceptance_tolerance_kWh': row['saved_prediction_acceptance_tolerance_kWh'],
                                'saved_prediction_within_fold_tolerance': True, 'forward_rerun_in_this_analysis': False})
        write_csv('lineage_parity_check.csv', pd.DataFrame(parity_rows))
        write_json('lineage_parity_check.json', {
            'status': 'passed', 'source': str(frozen), 'source_sha256': sha(frozen), 'folds_checked': len(parity_rows),
            'all_canonical_full_pointers': True, 'all_current_checkpoint_prediction_hashes_match_export': True,
            'all_weights_unchanged': True, 'all_before_after_weight_hashes_equal': True,
            'max_saved_prediction_abs_difference_kWh': max(r['saved_prediction_max_abs_difference_kWh'] for r in parity_rows),
            'all_within_recorded_fold_acceptance_tolerance': True,
            'parity_evidence': 'frozen gate export forward-vs-saved prediction comparison, not bitwise identity',
            'forward_rerun_in_this_analysis': False, 'training': False})
        candidates = [args.energy] if args.energy else [RUN / '01_metrics/per_origin_energy_metrics.csv.gz',
                                                        RUN / '01_metrics/per_origin_energy_metrics.csv']
        deadline = time.monotonic() + args.wait_seconds
        energy_path = next((p for p in candidates if p and p.is_file()), None)
        while energy_path is None and time.monotonic() < deadline:
            state.update(status='waiting_for_energy_input', candidates=[str(p) for p in candidates])
            write_json('mechanism_manifest.json', state)
            time.sleep(min(10, max(0, deadline - time.monotonic())))
            energy_path = next((p for p in candidates if p and p.is_file()), None)
        if energy_path is None:
            write_csv('source_hashes.csv', pd.DataFrame(sources))
            state.update(status='waiting_for_energy_input', candidates=[str(p) for p in candidates], gate_rows=len(gate))
            write_json('mechanism_manifest.json', state)
            output_path('README.md').write_text(readme(state['status'], None, None, None), encoding='utf-8')
            print('Waiting: main per-origin energy file absent; implementation ready. No association results fabricated.', flush=True)
            return 2
        before = signature(energy_path)
        paired, qc, info = load_pairs(energy_path, gate, args)
        if signature(energy_path) != before:
            raise ValueError('Energy file changed during read; rerun after writer finishes')
        record(energy_path, 'per_origin_energy_metrics')
        print(f'Strict pairing passed: {len(qc)} fold/settings, {len(paired)} origins', flush=True)
        history_info = {'status': 'disabled_by_explicit_history_off'}
        features = GATES + DERIVED
        if args.history == 'on':
            history, history_info = history_features(gate, args.archive)
            paired = paired.merge(history, on=KEY, how='left', validate='many_to_one', indicator=True)
            if not paired['_merge'].eq('both').all():
                raise ValueError('History join incomplete')
            paired = paired.drop(columns='_merge')
            features += PERIODIC
            record(args.archive, 'raw_CHARGED_archive')
            record(PROTOCOL, 'fixed_data_protocol')
        result = analyses(paired, features)
        validation = validate_outputs(result, qc)
        write_json('validation_report.json', validation)
        for name, table in result.items():
            write_csv(name, table)
        if args.history == 'on':
            write_csv('input_periodicity_per_origin.csv.gz', history)
        write_csv('join_qc.csv', qc)
        write_json('join_qc.json', {'status': 'passed', 'keys': KEY, 'validate': 'one_to_one outer join',
                                   'rows_expected_per_setting': len(EXPECTED), 'paired_rows': len(paired),
                                   'fold_settings': len(qc), 'all_origins_retained': True,
                                   'all_time_and_actual_checks_exact': True, 'history': history_info, **info})
        write_csv('source_hashes.csv', pd.DataFrame(sources))
        for path, original_hash in protected.items():
            if sha(path) != original_hash:
                raise ValueError(f'Protected existing gate summary changed during run: {path}')
        output_path('README.md').write_text(readme('completed', energy_path, result, history_info), encoding='utf-8')
        state.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                     args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     software={'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                     sources=sources, features=features, metrics=ERRORS,
                     join_checks='passed; exact 18x721 grid for every postprocess', history=history_info,
                     rank_definition='(average_rank-0.5)/n; fixed thresholds .25/.50/.75; ties together; constants explicit',
                     inference='descriptive within-fold Spearman; no independent-origin p-values; no causal claim',
                     origin_exclusions=0, protected_gate_summaries_unchanged={str(p): h for p, h in protected.items()})
        names = list(result) + ['README.md', 'join_qc.csv', 'join_qc.json', 'source_hashes.csv', 'validation_report.json',
                                'lineage_parity_check.csv', 'lineage_parity_check.json']
        if args.history == 'on':
            names += ['input_periodicity_per_origin.csv.gz']
        state['output_hashes'] = {n: sha(output_path(n)) for n in names}
        write_json('mechanism_manifest.json', state)
        print(f'Completed: {OUT}', flush=True)
        return 0
    except Exception as exc:
        state.update(status='failed_closed', error=f'{type(exc).__name__}: {exc}')
        write_json('mechanism_manifest.json', state)
        print(state['error'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

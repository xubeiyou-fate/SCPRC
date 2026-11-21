#!/usr/bin/env python3
"""Check parity to unchanged R12/review operation protocol; retain rule baselines."""
from pathlib import Path
import hashlib
import os
import json
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
SOURCE = Path(os.environ.get('SCPRC_OPERATIONAL_EVIDENCE_DIR', str(PROJECT_ROOT / 'experiments/support/review_evidence/07_Operational_Diagnostics/reanalysis'))).expanduser()
MAPPING = {'Original PhaseFormer': 'original_phaseformer', 'Exact-PCGrad': 'exact_pcgrad', 'Full SCPRC': 'full_scpcr'}


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f'Operational evidence is missing: {SOURCE}. Set SCPRC_OPERATIONAL_EVIDENCE_DIR to the supplied reanalysis directory.')
    src_ops = SOURCE / 'operational_metrics_per_fold.csv'
    src_cap = SOURCE / 'capacity_reserve_risk_per_fold.csv'
    ops = pd.read_csv(src_ops)
    cap = pd.read_csv(src_cap)
    new_ops = pd.read_csv(RUN / '01_metrics/per_fold_energy_metrics.csv')
    new_cap = pd.read_csv(HERE / 'capacity_per_fold_sensitivity.csv')
    rows = []
    comparisons = {'peak_level_mae': 'peak_magnitude_mae', 'peak_timing_mae_hours': 'peak_timing_mae_hours',
                   'ramp_mae': 'ramp_mae', 'total_volume_mae': 'total_24h_mae'}
    for old, new in MAPPING.items():
        a = ops[(ops.model == old) & (ops.layer == 'city_total')].set_index(['target_city', 'seed'])
        b = new_ops[(new_ops.model == new) & (new_ops.postprocess == 'nonnegative_clip')].set_index(['city', 'seed'])
        assert set(a.index) == set(b.index) and len(a) == 18
        for old_metric, new_metric in comparisons.items():
            delta = a[old_metric].sort_index().to_numpy() - b[new_metric].sort_index().to_numpy()
            rows.append({'scope': 'operation', 'model': new, 'metric': new_metric,
                         'max_abs_difference': float(np.abs(delta).max()), 'folds': len(delta)})
        a = cap[(cap.model == old) & (cap.safety_margin_pct == 10) & (cap.overload_penalty == 10)].set_index(['target_city', 'seed'])
        b = new_cap[(new_cap.model == new) & (new_cap.margin == .1) & (new_cap.penalty == 10)].set_index(['city', 'seed'])
        for om, nm in {'relative_cost_normalized_mean': 'total_proxy_normalized',
                       'overload_window_rate_pct': 'overload_window_rate_pct',
                       'overreserve_ratio_pct': 'excess_reservation_pct',
                       'overload_volume_mean': 'overload_volume'}.items():
            delta = a[om].sort_index().to_numpy() - b[nm].sort_index().to_numpy()
            rows.append({'scope': 'capacity', 'model': new, 'metric': nm,
                         'max_abs_difference': float(np.abs(delta).max()), 'folds': len(delta)})
    parity = pd.DataFrame(rows)
    parity.to_csv(HERE / 'legacy_parity_checks.csv', index=False)
    assert parity.max_abs_difference.max() < 1e-5, parity.to_string()
    for filename in ('operational_metrics_equal_fold.csv', 'capacity_reserve_risk_primary_equal_fold.csv'):
        src = SOURCE / filename
        (HERE / ('legacy_' + filename)).write_bytes(src.read_bytes())
    meta = {'parity_passed': True, 'max_abs_difference': float(parity.max_abs_difference.max()),
            'protocol': 'station-level max(raw,0), sum stations, frozen10%margin/penalty10; native units',
            'units': 'native hourly energy amount; no kW conversion or money',
            'denominator': 'realized 24h city peak; max(peak,1e-12)',
            'baselines_retained': ['Seasonal naive 24 h', 'Seasonal naive 168 h', 'Oracle actual peak'],
            'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (src_ops, src_cap)]}
    (HERE / 'parity_manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    (HERE / 'README.md').write_text('# 容量情景与旧协议复核\n\n'
        'Original、Exact、Full的新重算与已有review/R12的非负截断操作诊断逐fold核对，差异小于1e-5。'
        '季节24/168基线及oracle单独保留在legacy表中。新16模型的容量敏感性不混入raw指标。\n\n'
        'reserved=max预测城市24h序列×(1+margin)，overload=sum_h max(actual-reserved,0)，'
        'proxy=reserved+penalty×overload。归一化分母=max(actual_peak,1e-12)，逐origin归一后逐fold平均。'
        'reserve_normalized+penalty_normalized=total_proxy_normalized。excess_reservation只计高于实际峰值的预留。\n\n'
        'primary margin=10%，penalty=10；完整网格margin0/5/10/20%，penalty1/5/10/20。'
        '不把原始小时能量量值转成kW或货币。oracle使用已知实际峰值，只是不可部署的理想参照；未证明真实调度收益。\n', encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

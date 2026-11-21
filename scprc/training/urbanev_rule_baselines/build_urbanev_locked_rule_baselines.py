#!/usr/bin/env python3
"""Recompute no-training UrbanEV February rule baselines on locked seed17/29 windows.

Rules are defined by the pinned SCPRC-v2.2 runner. This reads the primary
volume.csv only to construct forecasts from the 168-hour context and score them
against the fixed February 24-hour targets. It never changes model/gate decisions.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
EXP = ROOT / 'scprc/training'
OUT = EXP / 'urbanev_locked_rule_baselines_20261022'
DATA = ROOT / 'research/urbanev_timemixer/sources/UrbanEV/data/volume.csv'
SEED_DIRS = {s: EXP / f'urbanev_external_scpcr_v22_volume_csv_multiseed_20261021_seed{s}' for s in (17,29)}
LEGACY = EXP / 'urbanev_external_scpcr_v22_20261020/result.json'
CONTEXT, HORIZON, VAL_END, TEST_END = 168, 24, 3672, 4344


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str,float]:
    e=pred.astype(np.float64)-truth.astype(np.float64)
    return {
        'macro_site_rmse': float(np.sqrt(np.mean(e*e,axis=(0,1))).mean()),
        'macro_site_mae': float(np.mean(np.abs(e),axis=(0,1)).mean()),
        'pooled_rmse': float(np.sqrt(np.mean(e*e))),
        'pooled_mae': float(np.mean(np.abs(e))),
    }

def pct(candidate: float, ref: float) -> float:
    return (candidate-ref)/ref*100.0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA)
    if raw.shape != (TEST_END,276): raise ValueError(f'unexpected CSV shape {raw.shape}')
    timestamps=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:,0],errors='raise'))
    expected=pd.date_range('2022-09-01 00:00:00',periods=TEST_END,freq='h')
    if not timestamps.equals(expected): raise ValueError('unexpected timestamp index')
    vals=raw.iloc[:,1:].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all() or (vals<0).any(): raise ValueError('invalid target values')
    starts=np.arange(VAL_END,TEST_END-HORIZON+1,dtype=np.int64)
    if len(starts)!=649: raise AssertionError(len(starts))
    truth=np.stack([vals[s:s+HORIZON] for s in starts])
    preds={'persistence_last_value':[], 'daily_mean_last_24h':[], 'weekly_seasonal_naive_168h_lag':[]}
    for s in starts:
        hist=vals[s-CONTEXT:s]
        preds['persistence_last_value'].append(np.repeat(hist[-1:,:],HORIZON,axis=0))
        preds['daily_mean_last_24h'].append(np.repeat(hist[-24:,:].mean(axis=0,keepdims=True),HORIZON,axis=0))
        # Matches pinned protocol: one-week-ago values for the target's same 24 hours.
        preds['weekly_seasonal_naive_168h_lag'].append(hist[-168:-144,:])
    pred_arrays={name:np.asarray(v,dtype=np.float32) for name,v in preds.items()}
    rule_scores={name:metrics(v,truth) for name,v in pred_arrays.items()}
    # Preserve paired origin-level arrays so later uncertainty summaries can be
    # computed without rerunning models or altering the locked split.
    archive={
        'february_starts':starts.astype(np.int64),
        'february_timestamps':np.asarray([str(timestamps[int(s)]) for s in starts],dtype=str),
        'site_ids':np.asarray(raw.columns[1:],dtype=str),
        'truth':truth.astype(np.float32),
    }
    archive.update(pred_arrays)
    for name,pred in pred_arrays.items():
        err=pred.astype(np.float64)-truth
        archive[name+'__window_rmse']=np.sqrt(np.mean(err*err,axis=(1,2))).astype(np.float32)
        archive[name+'__window_mae']=np.mean(np.abs(err),axis=(1,2)).astype(np.float32)
        total_err=err.sum(axis=2)
        archive[name+'__window_total_rmse']=np.sqrt(np.mean(total_err*total_err,axis=1)).astype(np.float32)
        archive[name+'__window_total_mae']=np.mean(np.abs(total_err),axis=1).astype(np.float32)
    np.savez_compressed(OUT/'locked_february_rule_predictions.npz',**archive)

    rows=[]
    for seed, d in SEED_DIRS.items():
        r=json.loads((d/'result.json').read_text())
        if r['seed'] != seed: raise ValueError(f'seed mismatch in {d}')
        if r['urban_volume']['sha256'] != sha256(DATA): raise ValueError(f'data hash mismatch seed {seed}')
        if r['scores']['february']['phaseformer_exact_source_scale']['windows'] != 649: raise ValueError('window mismatch')
        with np.load(d/'predictions.npz',allow_pickle=False) as archived:
            if not np.array_equal(archived['february_starts'],starts):
                raise ValueError(f'locked February origin mismatch for seed {seed}')
            expected_target_timestamps=np.asarray([str(timestamps[int(s)]) for s in starts],dtype=str)
            if not np.array_equal(archived['february_timestamps'],expected_target_timestamps):
                raise ValueError(f'locked February timestamp mismatch for seed {seed}')
        # Baseline recomputation is seed-invariant. Store one row per paired seed so
        # readers can align each rule with that seed's exact/candidate/release scores.
        for name, score in rule_scores.items():
            exact=r['scores']['february']['phaseformer_exact_source_scale']
            cand=r['scores']['february']['scpcr_v22_pcgrad_candidate_diagnostic_source_scale']
            strict=r['scores']['february']['scpcr_v22_pcgrad_strict_release_source_scale']
            rows.append({
                'seed':seed,'method':name,'windows':649,'sites':275,
                'macro_site_rmse':score['macro_site_rmse'],'macro_site_mae':score['macro_site_mae'],
                'pooled_rmse':score['pooled_rmse'],'pooled_mae':score['pooled_mae'],
                'exact_macro_site_rmse':exact['macro_rmse'],'exact_macro_site_mae':exact['macro_mae'],
                'candidate_macro_site_rmse':cand['macro_rmse'],'candidate_macro_site_mae':cand['macro_mae'],
                'strict_macro_site_rmse':strict['macro_rmse'],'strict_macro_site_mae':strict['macro_mae'],
                'delta_vs_exact_macro_rmse_pct':pct(score['macro_site_rmse'],exact['macro_rmse']),
                'delta_vs_exact_macro_mae_pct':pct(score['macro_site_mae'],exact['macro_mae']),
                'source_volume_sha256':sha256(DATA),
                'eval_start':str(timestamps[starts[0]]),'eval_end_label':str(timestamps[starts[-1]+HORIZON-1]),
            })
    # Include the fixed SCPRC rows in a compact aligned table from stored per-seed results.
    model_rows=[]
    for seed,d in SEED_DIRS.items():
        r=json.loads((d/'result.json').read_text())
        for key, label in [('phaseformer_exact_source_scale','PhaseFormer exact'),('scpcr_v22_pcgrad_candidate_diagnostic_source_scale','SCPRC candidate (diagnostic)'),('scpcr_v22_pcgrad_strict_release_source_scale','SCPRC strict release')]:
            s=r['scores']['february'][key]
            model_rows.append({'seed':seed,'method':label,'windows':649,'sites':275,
                'macro_site_rmse':s['macro_rmse'],'macro_site_mae':s['macro_mae'],
                'pooled_rmse':s['rmse'],'pooled_mae':s['mae'],
                'source_volume_sha256':sha256(DATA),'eval_start':str(timestamps[starts[0]]),
                'eval_end_label':str(timestamps[starts[-1]+HORIZON-1])})
    # Save a long-form table with candidate/model and rule entries; exact-relative
    # effects are included for all methods, while strict release tracks exact.
    byseed={s:json.loads((d/'result.json').read_text()) for s,d in SEED_DIRS.items()}
    exact_byseed={s:byseed[s]['scores']['february']['phaseformer_exact_source_scale'] for s in byseed}
    combined=[]
    for row in model_rows:
        seed=row['seed']; ex=exact_byseed[seed]
        row={**row,
             'delta_vs_exact_macro_rmse_pct':pct(row['macro_site_rmse'],ex['macro_rmse']),
             'delta_vs_exact_macro_mae_pct':pct(row['macro_site_mae'],ex['macro_mae'])}
        combined.append(row)
    combined.extend(rows)
    methods_order=['PhaseFormer exact','SCPRC candidate (diagnostic)','SCPRC strict release',*rule_scores.keys()]
    summary_rows=[]
    for method in methods_order:
        subset=[r for r in combined if r['method']==method]
        # Mean metrics across locked seeds; rule baselines repeat because they are seed-invariant.
        summary_rows.append({'seed':'mean(17,29)','method':method,'windows':649,'sites':275,
            'macro_site_rmse':float(np.mean([x['macro_site_rmse'] for x in subset])),
            'macro_site_mae':float(np.mean([x['macro_site_mae'] for x in subset])),
            'pooled_rmse':float(np.mean([x['pooled_rmse'] for x in subset])),
            'pooled_mae':float(np.mean([x['pooled_mae'] for x in subset])),
            'delta_vs_exact_macro_rmse_pct':float(np.mean([x['delta_vs_exact_macro_rmse_pct'] for x in subset])),
            'delta_vs_exact_macro_mae_pct':float(np.mean([x['delta_vs_exact_macro_mae_pct'] for x in subset])),
        })
    fields=['seed','method','windows','sites','macro_site_rmse','macro_site_mae','pooled_rmse','pooled_mae','delta_vs_exact_macro_rmse_pct','delta_vs_exact_macro_mae_pct','source_volume_sha256','eval_start','eval_end_label']
    with (OUT/'locked_february_results_by_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(combined)
    fields2=['seed','method','windows','sites','macro_site_rmse','macro_site_mae','pooled_rmse','pooled_mae','delta_vs_exact_macro_rmse_pct','delta_vs_exact_macro_mae_pct']
    with (OUT/'locked_february_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(summary_rows)

    legacy=json.loads(LEGACY.read_text())
    if legacy['urban_volume']['sha256'] != sha256(DATA): raise ValueError('legacy run used a different dataset file')
    legacy_scores=legacy['scores']['february']
    legacy_comp={k:{'macro_site_rmse':legacy_scores[k]['macro_rmse'],'macro_site_mae':legacy_scores[k]['macro_mae'],
                    'pooled_rmse':legacy_scores[k]['rmse'],'pooled_mae':legacy_scores[k]['mae']}
                 for k in ('weekly_naive','daily_mean','phaseformer_exact_source_scale','scpcr_v22_pcgrad_candidate_diagnostic_source_scale')}
    payload={
      'protocol':{'dataset':'UrbanEV Shenzhen volume.csv','role':'same-city cross-dataset/spatial-granularity diagnostic',
        'locked_seeds':[17,29],'split':'February 2023 held-out; 2023-02-01 00:00 through 2023-02-28 23:00 labels',
        'forecast_origins':649,'context_hours':168,'horizon_hours':24,'sites':275,
        'source_volume_sha256':sha256(DATA),'release_gate_modified':False,'training_performed':False,
        'test_labels_used_for_selection':False,'metrics':'macro_site_* = average of per-site metrics over 275 sites; pooled_* = all windows×hours×sites combined',
        'source_code':'run_urbanev_external_scprc.py naive_predictions()/score()'},
      'methods':{'persistence_last_value':'repeat most recent observed hourly value for all 24 forecast hours',
        'daily_mean_last_24h':'repeat the mean of the last 24 observed hours for all 24 forecast hours',
        'weekly_seasonal_naive_168h_lag':'copy the 24 observations exactly 168 hours before forecast start'},
      'per_seed_and_mean':combined+summary_rows,
      'legacy_seed13_metric_audit':{'seed':legacy['seed'],'same_data_hash':True,'same_windows':legacy_scores['weekly_naive']['windows'],
         'scores':legacy_comp,
         'note':'Legacy 2026-10-20 summary.md prints pooled RMSE. Locked multiseed headline uses macro-site RMSE. MAE is effectively identical under both aggregation schemes. This explains the apparent RMSE scale change; seed13 must remain separately labeled, not pooled into locked seeds 17/29.'}
    }
    (OUT/'locked_february_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')

    def find(method): return next(x for x in summary_rows if x['method']==method)
    lines=['# UrbanEV Shenzhen locked rule-baseline audit (February 2023)','',
      'This is a no-training add-on to the locked primary `volume.csv` same-city cross-dataset diagnostic. It reuses the exact February evaluation indices of the seed-17/29 runs (649 rolling 168→24 origins, 275 zones). Rule forecasts are computed only from each origin’s preceding 168 observed hours. The fixed support gate and release decisions were not changed; February labels were used only to score the fixed predictions.','',
      '## Protocol and aggregation','',
      '- Data: `research/urbanev_timemixer/sources/UrbanEV/data/volume.csv`; SHA-256: `'+sha256(DATA)+'`.','- Test labels: 2023-02-01 00:00–2023-02-28 23:00; 649 rolling origins, 275 zones.','- Context/horizon: 168 hours / 24 hours.','- Persistence: repeat the last observed hourly value; daily mean: repeat the mean of the last 24 observed hours; weekly seasonal naive: copy the same 24 hours from 168 hours earlier.','- `Macro RMSE` is the average of the 275 per-zone RMSE values; `MAE` averages over all zones, origins, and horizon points. `Pooled RMSE` is also included to make the metric aggregation explicit. Earlier locked multiseed headline values use macro-site RMSE.','- Rule-baseline scores are deterministic and therefore identical for seeds 17 and 29; they are repeated in the per-seed CSV only to align with the model rows.','',
      '## Locked results','',
      '| Method | Macro RMSE | MAE | Pooled RMSE | Δ macro RMSE vs Exact | Δ MAE vs Exact |','|---|---:|---:|---:|---:|---:|']
    for x in summary_rows:
      lines.append(f"| {x['method']} | {x['macro_site_rmse']:.3f} | {x['macro_site_mae']:.3f} | {x['pooled_rmse']:.3f} | {x['delta_vs_exact_macro_rmse_pct']:+.2f}% | {x['delta_vs_exact_macro_mae_pct']:+.2f}% |")
    lines += ['',
      '## Interpretation','',
      'On the locked, comparable macro-site metric, both no-training rules are much worse than PhaseFormer exact and SCPRC candidate; weekly seasonal naive is the strongest of the three rules, but still has substantially higher error. The SCPRC candidate is lower than Exact in both locked seeds, while the frozen support gate rejects it for both seeds; strict release therefore remains equal to Exact. This diagnostic supports candidate accuracy under this same-city dataset shift, but does not turn the candidate into a released gain or independent-city generalization claim.','',
      '## Seed-13 report reconciliation','',
      'The early single-seed `urbanev_external_scpcr_v22_20261020/summary.md` lists February RMSE values 322.241 (weekly naive), 257.995 (Exact), and 198.544 (candidate). Those are **pooled RMSE** values. The seed-13 `result.json` also contains the directly comparable macro-site RMSE values: 106.138, 90.929, and 71.015, respectively. The locked seed-17/29 summary uses macro-site RMSE (about 76–78), not pooled RMSE (about 204–224). MAE is the same to rounding under both aggregations. Thus the apparent RMSE scale difference comes from aggregation, not a new test window or a changed file. Seed 13 is an earlier single-seed exploratory run and remains separate from the preregistered locked seed17/29 replication.','',
      'Files: `locked_february_results_by_seed.csv`, `locked_february_summary.csv`, `locked_february_summary.json`.','']
    (OUT/'locked_february_report.md').write_text('\n'.join(lines))
    print(OUT)
    for x in summary_rows:
      print(x['method'],x['macro_site_rmse'],x['macro_site_mae'],x['pooled_rmse'],x['delta_vs_exact_macro_rmse_pct'],x['delta_vs_exact_macro_mae_pct'])

if __name__=='__main__': main()

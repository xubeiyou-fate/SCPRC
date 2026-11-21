#!/usr/bin/env python3
"""Score fixed CHARGED no-training rules on the frozen 18-fold August grid.

Reads only each city's prefix through 2023-09-01 (the August test end), uses the
preceding 168 observed hours for each rolling origin, and does no fitting.
"""
from __future__ import annotations
import csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'scprc/training').is_dir() and (p / "scprc").is_dir())
EXP=ROOT/'scprc/training'
PRED_ROOT=EXP/'charged_scpcr_v22_nested_source_calibration_20261021'
ARCHIVE=ROOT/'scprc/data_adapters/charged/Hourly.zip'
OUT=Path(__file__).resolve().parent
CITIES=('AMS','JHB','LOA','MEL','SPO','SZH')
SEEDS=(13,17,29)
STARTS=np.arange(2928,3649,dtype=np.int64)
H=24
N=721
BLOCK=168
REPS=5000
RNG=20261023
METHODS=('persistence_last','seasonal24_naive','seasonal168_naive','daily_mean_24h')

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read_city(city:str):
    # 3,672 rows end at 2023-09-01 00:00; no September outcome rows are read.
    with zipfile.ZipFile(ARCHIVE) as z:
        with z.open(f'{city}/volume.csv') as f:
            frame=pd.read_csv(f,nrows=3672)
    times=pd.DatetimeIndex(pd.to_datetime(frame.iloc[:,0],errors='raise'))
    expected=pd.date_range('2023-04-01 00:00:00',periods=3672,freq='h')
    if not times.equals(expected): raise ValueError(f'{city}: unexpected hourly grid')
    vals=frame.iloc[:,1:].to_numpy(dtype=np.float64)
    if vals.shape[0]!=3672 or not np.isfinite(vals).all(): raise ValueError(f'{city}: invalid values')
    return vals,tuple(str(x) for x in frame.columns[1:])

def score_city(city:str):
    vals,sites=read_city(city)
    errors={m:{'mse':np.empty(N), 'mae':np.empty(N)} for m in METHODS}
    for i,s in enumerate(STARTS):
        hist=vals[s-168:s]
        truth=vals[s:s+H]
        pred={
            'persistence_last':np.repeat(hist[-1:,:],H,axis=0),
            'seasonal24_naive':hist[-24:,:],
            'seasonal168_naive':hist[-168:-144,:],
            'daily_mean_24h':np.repeat(hist[-24:,:].mean(axis=0,keepdims=True),H,axis=0),
        }
        for m,p in pred.items():
            err=p-truth
            errors[m]['mse'][i]=np.mean(err*err)
            errors[m]['mae'][i]=np.mean(np.abs(err))
    return errors, len(sites), sites, vals

def pct(a,b): return (a/b-1.0)*100.0

def block_indices(rng):
    chunks=int(np.ceil(N/BLOCK))
    starts=rng.integers(0,N-BLOCK+1,size=(REPS,chunks))
    return (starts[...,None]+np.arange(BLOCK)).reshape(REPS,-1)[:,:N]

def bootstrap(rows):
    rng=np.random.default_rng(RNG)
    idx=block_indices(rng)
    out={}
    for method in METHODS:
        fold_draws={'rmse':[],'mae':[]}
        fold_points={'rmse':[],'mae':[]}
        for row in rows:
            exact=row['exact_origin']
            rule=row['rules'][method]
            # For this resampling replicate, paired blocks are sampled identically
            # for method and Exact within each city×seed fold.
            exact_mse=exact['mse'][idx].mean(axis=1); rule_mse=rule['mse'][idx].mean(axis=1)
            exact_mae=exact['mae'][idx].mean(axis=1); rule_mae=rule['mae'][idx].mean(axis=1)
            fold_draws['rmse'].append(np.sqrt(rule_mse/exact_mse)-1)
            fold_draws['mae'].append(rule_mae/exact_mae-1)
            fold_points['rmse'].append(np.sqrt(rule['mse'].mean()/exact['mse'].mean())-1)
            fold_points['mae'].append(rule['mae'].mean()/exact['mae'].mean()-1)
        # Equal seed weights within city, then equal city weights. Fold-specific
        # block draws are synchronized across folds and methods.
        for metric in ('rmse','mae'):
            arr=np.stack(fold_draws[metric],axis=0).reshape(len(CITIES),len(SEEDS),REPS)
            by_city=arr.mean(axis=1)
            rel=by_city.mean(axis=0)*100
            exact_scores=[]; rule_scores=[]
            for row in rows:
                exact=row['exact_origin']; rule=row['rules'][method]
                if metric=='rmse':
                    ex=np.sqrt(exact['mse'][idx].mean(axis=1)); ru=np.sqrt(rule['mse'][idx].mean(axis=1))
                else:
                    ex=exact['mae'][idx].mean(axis=1); ru=rule['mae'][idx].mean(axis=1)
                exact_scores.append(ex); rule_scores.append(ru)
            exmat=np.stack(exact_scores).reshape(len(CITIES),len(SEEDS),REPS).mean(axis=1).mean(axis=0)
            rumat=np.stack(rule_scores).reshape(len(CITIES),len(SEEDS),REPS).mean(axis=1).mean(axis=0)
            macro=(rumat/exmat-1)*100
            base={'macro_ratio_delta_pct':float((np.mean([r['scores']['macro'][method]['rule'][metric] for r in rows])/np.mean([r['scores']['macro'][method]['exact'][metric] for r in rows])-1)*100),
                  'macro_ratio_delta_ci95_pct':[float(x) for x in np.quantile(macro,[.025,.975])],
                  'macro_ratio_bootstrap_probability_better':float(np.mean(macro<0)),
                  'mean_fold_relative_delta_pct':float(np.mean(fold_points[metric])*100),
                  'mean_fold_relative_delta_ci95_pct':[float(x) for x in np.quantile(rel,[.025,.975])],
                  'mean_fold_relative_bootstrap_probability_better':float(np.mean(rel<0))}
            out.setdefault(method,{})[metric]=base
    return out

def main():
    rows=[]; raw={}
    for city in CITIES:
        scores,nsites,siteids,vals=score_city(city); raw[city]=scores
        for seed in SEEDS:
            pred_path=PRED_ROOT/f'outer_{city.lower()}_seed{seed}'/'outer_august_predictions.npz'
            freeze_path=PRED_ROOT/f'outer_{city.lower()}_seed{seed}'/'freeze.json'
            freeze=json.loads(freeze_path.read_text())
            if (freeze.get('status')!='frozen_before_outer_august_materialization'
                    or freeze.get('outer_city')!=city or freeze.get('seed')!=seed
                    or freeze.get('outer_august_labels_materialized') is not False):
                raise ValueError(f'not a valid pre-August freeze: {freeze_path}')
            with np.load(pred_path,allow_pickle=False) as z:
                starts=z['label_starts'].astype(np.int64)
                if not np.array_equal(starts,STARTS): raise ValueError(f'origin mismatch: {pred_path}')
                exact={'mse':z['phaseformer_exact_origin_mse'].astype(np.float64),
                       'mae':z['phaseformer_exact_origin_mae'].astype(np.float64)}
                # Cross-check the score labels against the frozen primary archive.
                if z['truth_raw'].shape[0]!=N or z['truth_raw'].shape[1]!=H or z['truth_raw'].shape[2]!=nsites:
                    raise ValueError(f'site/horizon mismatch {pred_path}')
                frozen_truth=z['truth_raw']
                if seed==SEEDS[0]:
                    for i,s in enumerate(STARTS):
                        if not np.array_equal(vals[s:s+H].astype(np.float32),frozen_truth[i]):
                            raise ValueError(f'CHARGED truth mismatch for {city}, origin index {int(s)}')
            rules={m:{'mse':raw[city][m]['mse'].copy(),'mae':raw[city][m]['mae'].copy()} for m in METHODS}
            point={}
            exact_mse=float(exact['mse'].mean()); exact_mae=float(exact['mae'].mean())
            for m in METHODS:
                rule_rmse=float(np.sqrt(rules[m]['mse'].mean())); rule_mae=float(rules[m]['mae'].mean())
                point[m]={'rmse':rule_rmse,'mae':rule_mae,'delta_rmse_pct':pct(rule_rmse,float(np.sqrt(exact_mse))),
                          'delta_mae_pct':pct(rule_mae,exact_mae)}
            rows.append({'city':city,'seed':seed,'sites':nsites,'exact_origin':exact,'rules':rules,
                         'scores':{'macro':{m:{'exact':{'rmse':float(np.sqrt(exact_mse)),'mae':exact_mae},'rule':point[m]} for m in METHODS}},
                         'prediction_path':str(pred_path),'prediction_sha256':sha256(pred_path),
                         'freeze_sha256':sha256(freeze_path),'point':point})
    # Require all 18 folds and the same three fixed seeds.
    if len(rows)!=18 or {(r['city'],r['seed']) for r in rows}!={(c,s) for c in CITIES for s in SEEDS}:
        raise RuntimeError('the expected 18 city×seed folds are incomplete')
    results=bootstrap(rows)
    fold_rows=[]
    for r in rows:
        for m in METHODS:
            fold_rows.append({'city':r['city'],'seed':r['seed'],'sites':r['sites'],'method':m,
                              'rmse':r['point'][m]['rmse'],'mae':r['point'][m]['mae'],
                              'delta_rmse_pct':r['point'][m]['delta_rmse_pct'],'delta_mae_pct':r['point'][m]['delta_mae_pct']})
    with (OUT/'charged_rule_baselines_18fold_by_fold.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fold_rows[0]));w.writeheader();w.writerows(fold_rows)
    summary_rows=[]
    for m in METHODS:
        for metric in ('rmse','mae'):
            scores=[r['point'][m][metric] for r in rows]
            summary_rows.append({'method':m,'metric':metric,'equal_fold_mean':float(np.mean(scores)),
                                 'fold_count':18,'cities':6,'seeds':'13,17,29',
                                 'macro_ratio_delta_vs_exact_pct':results[m][metric]['macro_ratio_delta_pct'],
                                 'mean_fold_relative_delta_vs_exact_pct':results[m][metric]['mean_fold_relative_delta_pct'],
                                 'macro_ratio_delta_ci95_pct':json.dumps(results[m][metric]['macro_ratio_delta_ci95_pct']),
                                 'mean_fold_relative_delta_ci95_pct':json.dumps(results[m][metric]['mean_fold_relative_delta_ci95_pct'])})
    with (OUT/'charged_rule_baselines_18fold_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    result={'status':'completed_read_only_rule_scoring','date':'2026-10-22','dataset':'CHARGED v3 six-city LOCO August 2023',
            'protocol':{'cities':list(CITIES),'seeds':list(SEEDS),'folds':18,'origins_per_fold':N,'origin_start_index':int(STARTS[0]),
                        'origin_end_index':int(STARTS[-1]),'horizon_hours':H,'context_hours':168,'block_origins':BLOCK,
                        'bootstrap_replicates':REPS,'bootstrap_seed':RNG,'raw_rows_read_per_city':3672,
                        'raw_rows_cutoff':'2023-09-01 00:00 exclusive; no September outcome rows','training':'none',
                        'rules':{'persistence_last':'repeat last observed value across horizon','seasonal24_naive':'copy same 24 hours from 24-hour lag',
                                 'seasonal168_naive':'copy same 24 hours from 168-hour lag',
                                 'daily_mean_24h':'repeat mean over preceding 24 hours'}},
            'data_archive_sha256':sha256(ARCHIVE),'prediction_and_freeze_hashes':{f"{r['city']}_seed{r['seed']}":{'prediction_sha256':r['prediction_sha256'],'freeze_sha256':r['freeze_sha256']} for r in rows},
            'metrics':{'RMSE':'sqrt pooled squared error over site×origin×horizon within fold; report is equal fold mean',
                       'MAE':'pooled absolute error over site×origin×horizon within fold; report is equal fold mean',
                       'paired_interval':'168-origin non-circular moving-block bootstrap; fixed cities/seeds; conditional temporal uncertainty'},
            'rule_comparisons_vs_exact':results,
            'fold_rows':fold_rows}
    (OUT/'charged_rule_baselines_18fold_summary.json').write_text(json.dumps(result,indent=2))
    md=['# CHARGED six-city 18-fold rule-baseline supplement','',
        'Read-only/no-training scoring on the same frozen August origins and city×seed folds as the primary nested experiment. For each origin, predictions use only the immediately preceding 168 observed hours. The pinned archive is read through 2023-09-01 00:00 exclusive; no September outcome rows are read. August is an already exposed retrospective development interval, so this is not blind confirmation.','',
        'The 168-origin paired moving-block intervals condition on the six fixed cities and existing model seeds. They quantify time-origin uncertainty only; cities and seeds are not resampled.','',
        'Point deltas and intervals use two estimands: macro-ratio is the ratio of equal-fold mean scores; mean fold-relative is the average of the 18 within-fold percentage changes. The two can differ substantially across cities with very different load scales.','',
        '| Rule | RMSE equal-fold mean | RMSE macro-ratio Δ [95% CI] | RMSE mean fold-relative Δ [95% CI] | MAE equal-fold mean | MAE macro-ratio Δ [95% CI] | MAE mean fold-relative Δ [95% CI] |','|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        rm=[r['point'][m]['rmse'] for r in rows]; ma=[r['point'][m]['mae'] for r in rows]
        rr=results[m]['rmse']; mm=results[m]['mae']
        rci=rr['macro_ratio_delta_ci95_pct']; rfi=rr['mean_fold_relative_delta_ci95_pct']
        mci=mm['macro_ratio_delta_ci95_pct']; mfi=mm['mean_fold_relative_delta_ci95_pct']
        md.append(f"| {m} | {np.mean(rm):.4f} | {rr['macro_ratio_delta_pct']:+.2f}% [{rci[0]:+.2f}, {rci[1]:+.2f}] | {rr['mean_fold_relative_delta_pct']:+.2f}% [{rfi[0]:+.2f}, {rfi[1]:+.2f}] | {np.mean(ma):.4f} | {mm['macro_ratio_delta_pct']:+.2f}% [{mci[0]:+.2f}, {mci[1]:+.2f}] | {mm['mean_fold_relative_delta_pct']:+.2f}% [{mfi[0]:+.2f}, {mfi[1]:+.2f}] |")
    md += ['', 'All four rules are scored on the identical 18 city×seed folds; they are substantially simpler than trained models, not compute-matched neural baselines. The intervals condition on the six cities and trained seeds. See the CSV for every fold and JSON for the reproducible protocol and hash manifest.']
    (OUT/'charged_rule_baselines_18fold_report.md').write_text('\n'.join(md)+'\n')
    print(OUT/'charged_rule_baselines_18fold_report.md')
    for m in METHODS:
        print(m, np.mean([r['point'][m]['rmse'] for r in rows]),np.mean([r['point'][m]['mae'] for r in rows]),results[m])

if __name__=='__main__': main()

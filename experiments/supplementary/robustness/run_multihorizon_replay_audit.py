#!/usr/bin/env python3
"""Retrospective horizon-prefix audit and frozen-input robustness feasibility check.

The current frozen NPZ archives contain 24-hour truth and predictions, so h=6,
12,24 metrics can be computed without new model inference. h=48 and corrupted-
input inference are deliberately blocked: no 48-hour outputs/checkpoints or
verified frozen input-cache/controller replay are available.
"""
from __future__ import annotations
import csv, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
FOLD_ROOT = ROOT / "experiments/supplementary/current_matched_ablation_20261027_homogeneous"
OUT = Path(__file__).resolve().parent
CITIES = ("AMS", "JHB", "LOA", "MEL", "SPO", "SZH")
SEEDS = (13, 17, 29)

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def metrics(yhat, y):
 e = yhat.astype(np.float64) - y.astype(np.float64)
 return float(np.sqrt(np.mean(e*e))), float(np.mean(np.abs(e)))

def main():
 rows=[]
 archive_records=[]
 for city in CITIES:
  for seed in SEEDS:
   p=FOLD_ROOT/f"target_{city.lower()}_seed{seed}/predictions.npz"
   if not p.is_file():
    rows.append({"city":city,"seed":seed,"status":"missing_archive"}); continue
   z=np.load(p)
   required=("truth_raw","pred_exact_pcgrad_raw","pred_scpcr_candidate_raw","label_starts")
   missing=[k for k in required if k not in z.files]
   if missing:
    rows.append({"city":city,"seed":seed,"status":"missing_keys:"+','.join(missing)}); continue
   y=z["truth_raw"]
   archive_records.append({"city":city,"seed":seed,"archive":str(p.relative_to(ROOT)),"sha256":sha256(p),"origins":int(y.shape[0]),"horizon_stored":int(y.shape[1]),"sites":int(y.shape[2])})
   for h in (6,12,24):
    for model,key in (("PhaseFormer Exact","pred_exact_pcgrad_raw"),("SCPRC Full","pred_scpcr_candidate_raw")):
     rmse,mae=metrics(z[key][:,:h],y[:,:h])
     rows.append({"city":city,"seed":seed,"horizon_hours":h,"model":model,"rmse":rmse,"mae":mae,"status":"retrospective_prefix_from_24h_archive"})
 # Aggregate by horizon/model over independent city x seed folds, equal weighted folds.
 summary=[]
 for h in (6,12,24):
  for model in ("PhaseFormer Exact","SCPRC Full"):
   sel=[r for r in rows if r.get("horizon_hours")==h and r.get("model")==model]
   summary.append({"horizon_hours":h,"model":model,"folds":len(sel),
                   "mean_fold_rmse":float(np.mean([r["rmse"] for r in sel])) if sel else None,
                   "mean_fold_mae":float(np.mean([r["mae"] for r in sel])) if sel else None})
 record={"status":"completed_horizon_prefix_audit_with_robustness_blockers","generated":"2026-10-30",
         "scope":"post-hoc h=6/12/24 prefix metrics from canonical homogeneous 24-hour predictions; no new model inference",
         "canonical_lineage":"experiments/supplementary/current_matched_ablation_20261027_homogeneous (canonical 18-fold matrix)",
         "superseded_lineage":"experiments/supplementary/current_scpcr_charged_20261024 (different lineage; prior h-prefix output is invalid for canonical claims)",
         "rows":rows,"summary":summary,"archives":archive_records,
         "blocked_experiments":{
          "horizon_48":"blocked: frozen checkpoints and prediction archives emit only 24 horizons; a 48-hour model must be retrained/evaluated under a predeclared protocol",
          "missing_input_robustness":"blocked: primary archives contain forecasts/labels but no exact model input histories, masks, scaler state, or verified frozen Online controller replay; perturbing cached predictions would not be a model-input experiment",
          "scale_shift_noise_spike":"blocked for same missing-input identity chain; raw Hourly.zip alone is insufficient to certify frozen-input reconstruction",
          "clean_match_gate":"required before any corruption replay: reproduce archived clean forecasts within predeclared tolerance and pass timestamp/site alignment checks"
         },
         "required_replay_outputs":["protocol.yaml","fold_manifest.json","input_cache.npz","mask_seed_manifest.json","clean_match_report.json","corrupted_predictions.npz","fold_metrics.csv"],
         "script_sha256":sha256(Path(__file__).resolve())}
 with (OUT/"multihorizon_prefix_metrics.csv").open('w',newline='',encoding='utf-8') as f:
  fields=["city","seed","horizon_hours","model","rmse","mae","status"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([r for r in rows if "horizon_hours" in r])
 (OUT/"multihorizon_prefix_summary.csv").write_text('horizon_hours,model,folds,mean_fold_rmse,mean_fold_mae\n'+'\n'.join(f"{r['horizon_hours']},{r['model']},{r['folds']},{r['mean_fold_rmse']},{r['mean_fold_mae']}" for r in summary)+'\n',encoding='utf-8')
 (OUT/"robustness_multihorizon_audit.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
 (OUT/"ROBUSTNESS_STATUS.md").write_text("""# Robustness and multi-horizon supplement status (2026-10-30)\n\n## Completed\n\nThe canonical homogeneous 24-hour prediction archives (`current_matched_ablation_20261027_homogeneous`) were audited and retrospectively sliced to h=6, 12, and 24 hours for Exact-PCGrad and SCPRC candidate. These are prefix metrics from existing forecasts, not new model runs. The earlier `current_scpcr_charged_20261024` lineage is superseded for canonical claims because its protocol differs. See `multihorizon_prefix_metrics.csv` and `multihorizon_prefix_summary.csv`.\n\n## Blocked until identity prerequisites are restored\n\n- h=48: current checkpoints and archives emit only 24 horizons.\n- Missingness, outage, scale-shift, noise, and spike tests: the primary archives do not contain exact model input histories, scaler state, masks, or a verified frozen Online controller replay. Raw `Hourly.zip` alone does not certify reconstruction.\n\nThe required clean-match gate is to reproduce archived clean forecasts from hash-matched code/checkpoints and pass timestamp/site alignment before any corruption replay. Do not report corrupted-input scores before that gate.\n""",encoding='utf-8')
 print(json.dumps({"status":record["status"],"rows":len(rows),"output":str(OUT)},ensure_ascii=False))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Train Exact-PCGrad and Full SCPRC with a genuine 48-hour output head."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve(); ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
for p in (ROOT,):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
legacy_name="calibration_protocol_constants"
if legacy_name not in sys.modules:
    legacy=types.ModuleType(legacy_name); legacy.CONTEXT=168; legacy.HORIZON=48; legacy.TRAIN_END=2928; sys.modules[legacy_name]=legacy
import experiments.commands.run_matched_ablation_fold as matched  # noqa: E402

SEEDS=(13,17,29); CITIES=matched.protocol.CITY_CODES

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def configure_h48():
    # The model builders and data window helpers use these module constants.
    matched.protocol.HORIZON=48
    matched.pcgrad.v21.HORIZON=48
    matched.pcgrad.v21.protocol.HORIZON=48

def run_fold(target:str,seed:int,*,output_root:Path,steps:int,device:str,torch_threads:int)->dict:
    configure_h48(); target=target.upper()
    out=output_root/f"target_{target.lower()}_seed{seed}"
    if (out/"result.json").exists(): raise FileExistsError(out/"result.json")
    out.mkdir(parents=True,exist_ok=True)
    cities,prov=matched.protocol.load_cities(matched.protocol.ARCHIVE)
    source=tuple(c for c in CITIES if c!=target); mean,std=matched.protocol.source_scaler(cities,source)
    starts,truth,values=matched.protocol._validation_data(cities[target]); n_sites=len(cities[target]["site_ids"])
    predictions={"label_starts":starts,"truth_raw":truth,"site_ids":np.asarray(cities[target]["site_ids"],dtype=str)}
    training={}; checkpoints={}; metrics={}; started=time.perf_counter()
    model,info=matched.train_arm("exact_pcgrad",cities,source,n_sites,mean,std,seed=seed,steps=steps,device=device,torch_threads=torch_threads)
    pred=matched.protocol._predict(model,values,starts,mean,std,device=device,batch_size=4).astype(np.float32)
    predictions["pred_exact_pcgrad_raw"]=pred; metrics["exact_pcgrad"]=matched.protocol._score(pred,truth); training["exact_pcgrad"]={**info,"metrics":metrics["exact_pcgrad"]}
    ck=out/"exact_pcgrad.pt"; torch.save({"arm":"exact_pcgrad","model_name":info["model"],"horizon":48,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},ck); checkpoints["exact_pcgrad"]=ck; del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model,info=matched.train_offline_full_candidate(cities,source,n_sites,mean,std,seed=seed,steps=steps,device=device,torch_threads=torch_threads)
    pred=matched.protocol._predict(model,values,starts,mean,std,device=device,batch_size=4).astype(np.float32)
    predictions["pred_scpcr_candidate_raw"]=pred; metrics["scpcr_candidate"]=matched.protocol._score(pred,truth); training["scpcr_candidate"]={**info,"metrics":metrics["scpcr_candidate"]}
    ck=out/"scpcr_candidate.pt"; torch.save({"arm":"scpcr_candidate","model_name":"scprc","horizon":48,"source_relation_threshold":info.get("source_relation_threshold"),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},ck); checkpoints["scpcr_candidate"]=ck; del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    predpath=out/"predictions.npz"; np.savez_compressed(predpath,**predictions)
    result={"experiment":"canonical h48 extension","status":"completed" if steps==2048 else "completed_smoke","target_city":target,"seed":seed,"source_cities":list(source),"protocol_id":"scpcr-matched-baseline-20261023-h48","protocol_file":str(matched.PROTOCOL_FILE),"protocol_sha256":sha256(matched.PROTOCOL_FILE),"retrospective":True,"blind_confirmation":False,"input":{"features":["volume"],"context_hours":168,"horizon_hours":48},"data_split":{"source_window":"2023-04-01 through 2023-07-31","evaluation_window":"2023-08-01 through 2023-08-31","evaluation_origins":int(len(starts))},"training_budget":{"source_minibatches_total":steps,"batch_size_windows":4,"optimizer":"AdamW","gradient_aggregation":"PCGrad"},"training":training,"metrics":metrics,"checkpoints":{k:str(v) for k,v in checkpoints.items()},"predictions":str(predpath),"prediction_sha256":sha256(predpath),"data_provenance":prov,"wall_seconds":time.perf_counter()-started}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--target",choices=CITIES,required=True); p.add_argument("--seed",type=int,choices=SEEDS,required=True); p.add_argument("--steps",type=int,default=2048); p.add_argument("--device",default="cuda"); p.add_argument("--torch-threads",type=int,default=2); p.add_argument("--output-root",type=Path,default=ROOT/"experiments/forecasting/h48")
    a=p.parse_args(); r=run_fold(a.target,a.seed,output_root=a.output_root,steps=a.steps,device=a.device,torch_threads=a.torch_threads); print(json.dumps({"status":r["status"],"target_city":a.target,"seed":a.seed,"shapes":{k:v for k,v in r["metrics"].items()}},sort_keys=True))
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Independent canonical-PCGrad controls; original trainer is called unchanged."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, importlib.util, json, os, platform, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
ANCHOR = ROOT / 'experiments/forecasting/h24_framework/run_canonical_anchor_fold.py'
spec = importlib.util.spec_from_file_location('supplement_canonical_anchor', ANCHOR)
anchor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anchor)
protocol, pcgrad = anchor.protocol, anchor.pcgrad
VARIANTS = ('full_same_run', 'phase_shared_strict', 'stationwise_matched', 'concat_project')

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n')
    os.replace(tmp, path)

def sha(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def tensor_hash(items):
    h=hashlib.sha256()
    for k,v in sorted(items):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def source_hashes():
    result={}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and str(p).startswith(str(ROOT)) and Path(p).is_file() and str(p).endswith('.py'):
            result[str(Path(p).resolve().relative_to(ROOT))]=sha(p)
    for p in (Path(__file__), HERE/'model_variants.py', ANCHOR, anchor.PROTOCOL_FILE):
        if p.is_file():result[str(p.relative_to(ROOT))]=sha(p)
    return result

def prepare_cache():
    cache=HERE/'data_cache';cache.mkdir(exist_ok=True)
    with open(cache/'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (cache/'metadata.json').exists():return
        cities, provenance=protocol.load_cities(protocol.ARCHIVE)
        meta={'provenance':provenance,'archive_sha256':sha(protocol.ARCHIVE),'cities':{}}
        for city,c in cities.items():
            values_path=cache/f'{city}_values.npy';np.save(values_path,c['values'])
            starts,truth,_=protocol._validation_data(c)
            target_time=np.asarray(c['times'])[starts[:,None]+np.arange(24)[None,:]]
            origin_time=np.asarray(c['times'])[starts-1]
            data=cache/f'{city}_fold_data.npz'
            np.savez(data,truth=truth,truth_raw=truth,label_starts=starts,site_ids=np.asarray(c['site_ids'],dtype=str),origin_time=origin_time,target_time=target_time)
            meta['cities'][city]={'site_ids':list(c['site_ids']),'values_sha256':sha(values_path),'fold_data_sha256':sha(data),'rows':int(len(c['values'])),'stations':len(c['site_ids']),'prefix_sha256':c['prefix_sha256']}
        atomic_json(cache/'metadata.json',meta)

def cached_cities():
    prepare_cache();cache=HERE/'data_cache';meta=json.loads((cache/'metadata.json').read_text())
    cities={c:{'values':np.load(cache/f'{c}_values.npy',mmap_mode='r'),'site_ids':tuple(m['site_ids'])} for c,m in meta['cities'].items()}
    return cities,meta

def prepare_fold(target, seed, output_root, device):
    cities,meta=cached_cities()
    sources=tuple(c for c in protocol.CITY_CODES if c!=target)
    fold=Path(output_root)/f'target_{target.lower()}_seed{seed}';fold.mkdir(parents=True,exist_ok=True)
    link=fold/'fold_data.npz'
    if not link.exists():link.symlink_to(HERE/'data_cache'/f'{target}_fold_data.npz')
    with open(fold/'.prepare.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        config=fold/'fold_protocol.json'
        if config.exists(): cfg=json.loads(config.read_text())
        else:
            mean,std=protocol.source_scaler(cities,sources)
            threshold=pcgrad.v21.cert._estimate_threshold(cities,sources,mean,std,seed=seed,device=device,quantile=.90)
            cfg={'target_city':target,'seed':seed,'source_cities':sources,'source_scaler':{'mean':mean,'std':std},'source_relation_threshold':float(threshold),'threshold_quantile':.9,'threshold_rng_seed':seed+7001,'data_provenance':meta['provenance'],'archive_sha256':meta['archive_sha256'],'fold_data_sha256':meta['cities'][target]['fold_data_sha256'],'context':168,'horizon':24,'source_train_rows':[0,2928],'evaluation_label_starts':[2928,3649],'evaluation_origin_count':721,'target_last_label_time':'2023-08-31 23:00:00','target_support_projection':False,'target_release_gate':False,'support_scale':1.0,'retrospective':True,'september_materialized':False}
            atomic_json(config,cfg)
    return cities,cfg,fold

def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

class Observer:
    """Observe loss, sampling, projection order and optimizer steps without changing math."""
    def __init__(self,out,steps,source_ids,cities):
        self.out=out;self.rounds=(steps+len(source_ids)-1)//len(source_ids);self.steps=0;self.losses=[];self.traces=[];self.orders=[];self.optimizer=None;self.started=time.perf_counter();self.last=self.started
        self.ptrs={cities[c]['values'].ctypes.data:c for c in source_ids}
    def __enter__(self):
        outer=self
        self.old_adam=torch.optim.AdamW;self.old_loss=protocol._objective;self.old_batch=protocol._make_batch;self.old_proj=pcgrad._project_conflicts
        class AdamW(self.old_adam):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);outer.optimizer=self
            def step(self,*a,**kw):
                result=super().step(*a,**kw);outer.steps+=1
                if outer.steps==1 or outer.steps%10==0 or outer.steps==outer.rounds:
                    now=time.perf_counter();record={'status':'training','optimizer_step':outer.steps,'optimizer_steps_expected':outer.rounds,'source_minibatches_seen':len(outer.traces),'elapsed_seconds':now-outer.started,'recent_loss_mean':float(np.mean(outer.losses[-5:])),'updated_at_unix':time.time(),'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
                    atomic_json(outer.out/'training_progress.json',record)
                    with open(outer.out/'training_curve.jsonl','a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                return result
        def loss(pred,y):
            v=self.old_loss(pred,y);outer.losses.append(float(v.detach()));return v
        def batch(values,starts,mean,std,rng,batch_size):
            # A copied generator predicts and records the actual unchanged sampler draw.
            probe=copy.deepcopy(rng);picks=probe.choice(starts,size=batch_size,replace=True)
            result=self.old_batch(values,starts,mean,std,rng,batch_size)
            outer.traces.append({'city':outer.ptrs.get(values.ctypes.data,'unknown'),'label_starts':picks.tolist()})
            outer.sampler_state=copy.deepcopy(rng.bit_generator.state)
            return result
        def project(grads,order):
            outer.orders.append(np.asarray(order).tolist());return self.old_proj(grads,order)
        torch.optim.AdamW=AdamW;protocol._objective=loss;protocol._make_batch=batch;pcgrad._project_conflicts=project
        return self
    def __exit__(self,*exc):
        torch.optim.AdamW=self.old_adam;protocol._objective=self.old_loss;protocol._make_batch=self.old_batch;pcgrad._project_conflicts=self.old_proj

def diagnostic_probe(model,values,starts,mean,std,device):
    selected=np.unique(np.minimum(np.array([0,24,168,336,504,672,720]),len(starts)-1))
    history=np.stack([values[s-168:s] for s in starts[selected]]).astype(np.float32)
    out={};captured={}
    def relation_hook(module,inputs,outputs):
        captured['z']=inputs[0].detach();captured['mixed']=outputs[0].detach()
    hook=model.relation.register_forward_hook(relation_hook)
    with torch.inference_mode():
        x=torch.from_numpy((history-mean)/std).to(device)
        try:_,diag=model(x,return_diagnostics=True)
        finally:hook.remove()
        _,stats=model.backbone.revin.normalize(x)
        scale=stats[1].squeeze(1).float()
        a=model.station_scale_gate(scale)
        disagreement=diag['relation_disagreement']
        b=(disagreement.median(dim=1,keepdim=True).values/disagreement.clamp_min(1e-6)).clamp(max=1.)
        bs=(torch.as_tensor(model.source_relation_threshold,device=device)/diag['source_peer_disagreement'].clamp_min(1e-6)).clamp(max=1.)
        raw_before=(captured['mixed']-captured['z'])*scale[:,:,None,None]
        raw_attenuated=raw_before*a[:,:,None,None]*b[:,:,None,None]*bs[:,None,None,None]
        raw_centered=raw_attenuated-raw_attenuated.mean(dim=1,keepdim=True)
        diag.update(station_scale=scale,scale_attenuation=a,peer_attenuation=b,source_envelope_attenuation=bs,raw_correction_before_attenuation=raw_before.abs().mean(dim=(-1,-2)),raw_correction_after_attenuation=raw_attenuated.abs().mean(dim=(-1,-2)),raw_correction_after_last_centering=raw_centered.abs().mean(dim=(-1,-2)))
    for key,value in diag.items():
        if torch.is_tensor(value):
            a=value.detach().float().cpu().numpy()
            if np.isfinite(a).all():out[key]={'shape':list(a.shape),'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'q':[float(v) for v in np.quantile(a,[0,.1,.5,.9,1])],'fraction_lt_1':float((a<1).mean()),'fraction_lt_0_9':float((a<.9).mean()),'fraction_lt_0_5':float((a<.5).mean())}
    return {'selection_rule':'origin indices [0,24,168,336,504,672,720], fixed before results','origin_indices':selected.tolist(),'diagnostics':out}

def run(args):
    from experiments.design_controls.model_variants import build_model
    if args.target not in protocol.CITY_CODES or args.seed not in (13,17,29):raise ValueError('outside city/seed matrix')
    if args.steps!=2048 and Path(args.output_root).resolve()==(HERE/'runs').resolve():raise ValueError('smoke must use separate output root')
    torch.set_num_threads(args.torch_threads)
    cities,cfg,fold=prepare_fold(args.target,args.seed,args.output_root,args.device)
    out=fold/args.variant
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text())
        if old.get('status')=='completed' and old.get('steps')==args.steps and sha(out/'prediction_raw.npy')==old['predictions_sha256']:
            print(json.dumps({'reused_completed':str(out)}));return
        raise FileExistsError(f'result exists, refusing overwrite: {out}')
    if out.exists() and any(out.iterdir()):
        archived=HERE/'interrupted_attempts'/f'{fold.name}_{args.variant}_{time.time_ns()}'
        archived.parent.mkdir(exist_ok=True);out.rename(archived)
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();mean=cfg['source_scaler']['mean'];std=cfg['source_scaler']['std'];sources=tuple(cfg['source_cities'])
    model=build_model(args.variant,len(cities[args.target]['site_ids']),cfg['source_relation_threshold'],seed=args.seed).float().to(args.device)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    hashes=source_hashes()
    initial={'all_state_sha256':tensor_hash(model.state_dict().items()),'backbone_state_sha256':tensor_hash(model.backbone.state_dict().items()),'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    atomic_json(out/'initialization.json',initial);atomic_json(out/'code_hashes.json',hashes)
    atomic_json(out/'run_config.json',{'variant':args.variant,'steps':args.steps,'fold':cfg,'source_hashes':hashes,'precision':'float32; no AMP','resume_policy':'completed runs only; interrupted arm restarts from seed, old attempt retained'})
    try:
        with Observer(out,args.steps,sources,cities) as obs:
            info=pcgrad._train_candidate_pcgrad(model,cities,sources,mean,std,seed=args.seed,steps=args.steps,device=args.device,torch_threads=args.torch_threads)
        if len(obs.traces)!=args.steps or obs.steps!=((args.steps+4)//5):raise AssertionError('actual budget mismatch')
        if any(t['city']=='unknown' for t in obs.traces):raise AssertionError('unidentified source batch')
        atomic_json(out/'sampling_trace.json',{'source_batches':obs.traces,'pcgrad_orders':obs.orders})
        atomic_json(out/'training_losses.json',{'loss_per_source_minibatch':obs.losses})
        prediction_start=time.perf_counter()
        starts=np.arange(2928,3672-24+1,dtype=np.int64)
        values=cities[args.target]['values']
        prediction=protocol._predict(model,values,starts,mean,std,device=args.device,batch_size=4).astype(np.float32)
        inference_seconds=time.perf_counter()-prediction_start
        with np.load(fold/'fold_data.npz',allow_pickle=False) as z:truth=z['truth'];expected_starts=z['label_starts']
        if not np.array_equal(starts,expected_starts) or prediction.shape!=truth.shape or not np.isfinite(prediction).all():raise AssertionError('prediction alignment/nonfinite error')
        tmp=out/'prediction_raw.tmp.npy';np.save(tmp,prediction);os.replace(tmp,out/'prediction_raw.npy')
        diagnostics=diagnostic_probe(model,values,starts,mean,std,args.device);atomic_json(out/'diagnostics.json',diagnostics)
        checkpoint={'variant':args.variant,'arm':args.variant,'state_dict':cpu_tree(model.state_dict()),'optimizer_state_dict':cpu_tree(obs.optimizer.state_dict()),'fold_protocol':cfg,'initialization':initial,'source_hashes':hashes,'training':info,'python_rng_state':random.getstate(),'numpy_global_rng_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'sampler_rng_state':getattr(obs,'sampler_state',None),'resume_policy':'completed arms only; no step-resume implementation'}
        torch.save(checkpoint,out/'checkpoint.tmp.pt');os.replace(out/'checkpoint.tmp.pt',out/'checkpoint.pt')
        result={'status':'completed' if args.steps==2048 else 'completed_smoke','variant':args.variant,'arm':args.variant,'target_city':args.target,'seed':args.seed,'steps':args.steps,'configured_source_minibatches':args.steps,'window_exposures':args.steps*4,'optimizer_steps':obs.steps,'protocol_id':anchor.PROTOCOL_ID,'protocol_sha256':sha(anchor.PROTOCOL_FILE),'training':info,'metrics':protocol._score(prediction,truth),'fold_protocol':cfg,'initialization':initial,'parameter_count':initial['parameter_count'],'source_hashes':hashes,'checkpoint':str(out/'checkpoint.pt'),'predictions':str(out/'prediction_raw.npy'),'checkpoint_sha256':sha(out/'checkpoint.pt'),'predictions_sha256':sha(out/'prediction_raw.npy'),'inference_seconds':inference_seconds,'wall_seconds':time.perf_counter()-started,'completed_at_unix':time.time(),'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu','cuda':torch.version.cuda,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}}
        atomic_json(out/'result.json',result);atomic_json(out/'training_progress.json',{'status':result['status'],'optimizer_step':obs.steps,'optimizer_steps_expected':obs.steps,'wall_seconds':result['wall_seconds'],'updated_at_unix':time.time()})
        print(json.dumps({'completed':str(out),'metrics':result['metrics'],'wall_seconds':result['wall_seconds']}),flush=True)
    except BaseException as e:
        atomic_json(out/'error.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target',type=str.upper,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--variant',choices=VARIANTS,required=True)
    ap.add_argument('--steps',type=int,default=2048);ap.add_argument('--device',default='cuda');ap.add_argument('--torch-threads',type=int,default=2);ap.add_argument('--output-root',type=Path,default=HERE/'runs')
    run(ap.parse_args())

"""Pinned UrbanEV protocols with explicit target indices and split boundaries."""
from __future__ import annotations
import hashlib, importlib.util
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "scprc/training").is_dir() and (parent / "scprc/models").is_dir())
DATA = ROOT / 'research/urbanev_timemixer/sources/UrbanEV/data'
PROTOCOL_VERSION = 'boundary-audited-v3'
LEGACY_PROTOCOL_VERSION = 'boundary-audited-v4'
DEV_PROTOCOL_VERSION = 'forward-dev-v1'


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA/name,index_col=0)


def _windows(values, starts, seq_len, horizon, lead_time=1, official_drop_last=False):
    starts=np.asarray(starts,dtype=np.int64)
    ends=starts+seq_len+lead_time-1+horizon
    valid=(starts>=0)&(ends<=len(values))
    if official_drop_last: valid &= ends<len(values)
    starts=starts[valid]
    if not len(starts):
        return np.empty((0,seq_len,values.shape[1]),np.float32),np.empty((0,horizon,values.shape[1]),np.float32)
    x=np.stack([values[s:s+seq_len] for s in starts]).astype(np.float32)
    y=np.stack([values[s+seq_len+lead_time-1:s+seq_len+lead_time-1+horizon] for s in starts]).astype(np.float32)
    return x,y


def _legacy_capacity(inf, columns):
    d=dict(zip(inf.TAZID.astype(str),inf.charge_count))
    return np.array([d[c] for c in columns],dtype=np.float64)


def _audited_capacity(inf, columns):
    d=inf.assign(TAZID=inf.TAZID.astype(str)).groupby('TAZID').charge_count.sum()
    return np.array([d[c] for c in columns],dtype=np.float64)


def _fold_bounds(time,fold):
    months=list(time.month.unique())
    if not 1<=fold<=len(months): raise ValueError('invalid fold')
    n=int(time.month.isin(months[:fold]).sum())
    tr=int(n*.8);va=int(tr+n*.1)
    return tr,va,n


def _time_marks(times):
    # Exactly equivalent to utils.timefeatures.time_features(...,freq='h').T.
    return np.column_stack([times.hour/23-.5,times.dayofweek/6-.5,
                            (times.day-1)/30-.5,(times.dayofyear-1)/365-.5]).astype(np.float32)


def prepare(config:dict[str,Any])->dict[str,Any]:
    protocol=config.get('protocol','research_volume')
    legacy=protocol.startswith('legacy_')
    transformer=protocol=='legacy_transformer_occ'
    target=config.get('target','occupancy' if legacy else 'volume')
    length=int(config.get('seq_len',12 if legacy else 168))
    horizon=int(config.get('horizon',config.get('pred_len',1 if legacy else 24)))
    lead=int(config.get('lead_time',1));fold=int(config.get('fold',6))
    audited=bool(config.get('audited',not legacy))
    df=_load_csv({'occupancy':'occupancy.csv','volume':'volume.csv','duration':'duration.csv'}[target])
    times=pd.DatetimeIndex(pd.to_datetime(df.index));columns=list(df.columns.astype(str))
    inf=pd.read_csv(DATA/'inf.csv')
    capacity=(_audited_capacity if audited else _legacy_capacity)(inf,columns)
    values=df.to_numpy(dtype=np.float64)
    if target=='occupancy': values=values/capacity
    if not np.isfinite(values).all(): raise ValueError('nonfinite source data')
    if transformer:
        # Official preprocess labels column0 OT and Dataset_Custom moves OT last.
        order=list(range(1,len(columns)))+[0]
        values=values[:,order];capacity=capacity[order];columns=[columns[i] for i in order]
    if legacy:
        tr,va,te=_fold_bounds(times,fold)
        if transformer: va=te-int(te*.1)
        mean=np.zeros(values.shape[1]);std=np.ones(values.shape[1]);scaled=values
    else:
        if protocol=='research_dev':
            dev_fold=str(config.get('dev_fold','jan')).lower()
            if dev_fold not in {'dec','jan'}:
                raise ValueError('research_dev dev_fold must be dec or jan')
            train_end,val_end=({'dec':('2022-12-01','2023-01-01'),
                                'jan':('2023-01-01','2023-02-01')})[dev_fold]
            tr=int((times<pd.Timestamp(train_end)).sum())
            va=int((times<pd.Timestamp(val_end)).sum());te=va
        elif protocol=='research_volume':
            tr=int((times<pd.Timestamp('2023-01-01')).sum())
            va=int((times<pd.Timestamp('2023-02-01')).sum());te=len(times)
        else:
            raise ValueError(f'unknown research protocol: {protocol}')
        mean=values[:tr].mean(0);std=values[:tr].std(0);std[std<1e-6]=1
        scaled=(values-mean)/std
    bounds={'train':(0,tr),'val':(tr,va)}
    if protocol!='research_dev': bounds['test']=(va,te)
    output={};label_indices={};window_starts={}
    raw_splits={k:scaled[a:b].copy() for k,(a,b) in bounds.items()}
    needs_marks = transformer or bool(config.get('use_time_marks', False))
    marks=_time_marks(times) if needs_marks else None
    for name,(a,b) in bounds.items():
        local_windows=legacy and not transformer
        lo=a if local_windows or a==0 else max(0,a-length-lead+1)
        # Official legacy ``create_rnn_data`` drops the final endpoint;
        # transformer Dataset_Custom retains it. ``arange`` has an exclusive
        # stop, hence the explicit protocol distinction.
        stop=b-length-lead-horizon+2-int(local_windows)
        starts=np.arange(lo,max(lo,stop),dtype=np.int64)
        x,y=_windows(scaled,starts,length,horizon,lead)
        yi=starts[:,None]+length+lead-1+np.arange(horizon)[None,:]
        assert len(x)==len(starts)
        assert not len(yi) or ((yi>=a)&(yi<b)).all(),(protocol,name,a,b,yi.min(),yi.max())
        output[name+'_x']=x;output[name+'_y']=y
        label_indices[name]=yi;window_starts[name]=starts
        if needs_marks:
            output[name+'_x_mark']=np.stack([marks[s:s+length] for s in starts]).astype(np.float32)
    if legacy and config.get('model') in {'lo','ar','arima'}:
        # The official statistics branch starts at L+h, one hour after neural
        # labels. Preserve this separately instead of silently aligning it.
        for name in ('val','test'):
            a,b=bounds[name];idx=np.arange(a+length+lead,b,dtype=np.int64)
            output[name+'_y']=scaled[idx,None,:]
            label_indices[name]=idx[:,None]
    # Include the complete pinned UrbanEV data snapshot in the provenance
    # digest.  The forecasting protocol only consumes a subset, but a run
    # must be invalidated if any companion data file is replaced.
    digest=hashlib.sha256()
    for path in sorted(DATA.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(DATA)).encode())
            digest.update(path.read_bytes())
    summary={k:{'bounds':list(bounds[k]),'samples':len(label_indices[k]),
                'first_label':int(label_indices[k].min()),'last_label':int(label_indices[k].max())}
             for k in bounds}
    # Static zone descriptors are available before the forecast origin.  They
    # are kept separate from the target windows so a model can use them as a
    # node chart without leaking future measurements.
    inf_static = pd.read_csv(DATA/'inf.csv')
    inf_static['TAZID'] = inf_static['TAZID'].astype(str)
    static_map = {}
    for taz, group in inf_static.groupby('TAZID'):
        weights = np.maximum(group['charge_count'].to_numpy(dtype=np.float64), 1e-8)
        static_map[taz] = np.asarray([
            weights.sum(), float(len(group)),
            np.average(group['longitude'], weights=weights),
            np.average(group['latitude'], weights=weights),
            float(group['area'].iloc[0]), float(group['perimeter'].iloc[0]),
        ], dtype=np.float64)
    static_raw = np.asarray([static_map[str(taz)] for taz in columns], dtype=np.float64)
    # Log heavy-tailed scale descriptors, then standardize over zones only.
    static_raw[:, [0, 1, 4, 5]] = np.log1p(np.maximum(static_raw[:, [0, 1, 4, 5]], 0.0))
    static_features = (static_raw - static_raw.mean(0)) / np.maximum(static_raw.std(0), 1e-8)
    output.update(adj=pd.read_csv(DATA/'adj.csv').to_numpy(dtype=np.float32),
                  distance=pd.read_csv(DATA/'distance.csv').to_numpy(dtype=np.float32),
                  scaler_mean=mean,scaler_std=std,capacity=capacity,
                  static_features=static_features.astype(np.float32), raw_splits=raw_splits,
                  label_indices=label_indices,window_starts=window_starts,
                  metadata={'protocol':protocol,'protocol_version':LEGACY_PROTOCOL_VERSION if legacy else DEV_PROTOCOL_VERSION if protocol=='research_dev' else PROTOCOL_VERSION,'target':target,
                            'audited':audited,'fold':fold,'seq_len':length,'horizon':horizon,
                            'dev_fold':str(config.get('dev_fold','jan')).lower() if protocol=='research_dev' else None,
                            'lead_time':lead,'time_start':str(times[0]),'time_end':str(times[-1]),
                            'data_sha256':digest.hexdigest(),'splits':summary,'TAZ_order':columns,
                            'supervision':'continuous' if horizon>1 else 'endpoint',
                            'statistical_legacy_one_hour_shift':legacy and config.get('model') in {'lo','ar','arima'}})
    return output


def restore(arr,data):
    if data['metadata']['protocol'].startswith('legacy_'):return arr
    return (np.asarray(arr,dtype=np.float64)*data['scaler_std'][None,None,:]+data['scaler_mean'][None,None,:]).astype(np.float32)

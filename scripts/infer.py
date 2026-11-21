#!/usr/bin/env python3
"""Portable inference using local, manifest-verified weights and original constructors.

Input/output are external .npy files. Histories are raw [B,168,N] volume;
CHARGED's two-variable DLinear also requires matching raw duration history.
No training, dataset discovery, network access or source-server import fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
_IMPORT_ORIGINS = {}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path,
                   help='Local checkpoint listed in manifests/CHECKPOINTS.csv')
    p.add_argument('--input', required=True, type=Path, help='External raw volume .npy [B,168,N]')
    p.add_argument('--output', required=True, type=Path, help='External output .npy [B,H,N], raw units')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--duration-input', type=Path, help='External raw duration .npy [B,168,N]; required for CHARGED DLinear')
    p.add_argument('--clip-min', type=float, default=None, help='Explicit output lower bound; default: no clipping')
    p.add_argument('--clip-max', type=float, default=None, help='Explicit output upper bound; default: no clipping')
    p.add_argument('--batch-size', type=int, default=16, help='Forward batch size (site windows for independent baselines)')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--metadata', type=Path, default=ROOT / 'manifests/INFERENCE_METADATA.json', help='Public inference metadata JSON bundled with the repository')
    return p


def inside(path, parent=ROOT):
    return Path(path).resolve().is_relative_to(parent.resolve())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def configure_imports():
    """Keep dependency installations and stdlib; remove all other project paths."""
    stdlib = Path(sysconfig.get_paths()['stdlib']).resolve()
    kept = []
    for value in sys.path:
        path = Path(value or '.').resolve()
        if inside(path) or path.is_relative_to(stdlib) or 'site-packages' in path.parts or 'dist-packages' in path.parts or (path.suffix == '.zip' and path.parent == stdlib.parent):
            kept.append(str(path))
    local = [ROOT, ROOT / 'scprc',
             ROOT / 'scprc/baselines', ROOT / 'scprc/training',
             ROOT / 'scprc/data_adapters', ROOT / 'scprc/data_adapters/charged']
    sys.path[:] = list(dict.fromkeys(map(str, local))) + [p for p in kept if p not in map(str, local)]
    verify_imports()


def verify_imports():
    stdlib = Path(sysconfig.get_paths()['stdlib']).resolve()
    local, dependencies, invalid = {}, set(), {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename or str(filename).startswith('<'):
            continue
        cached = _IMPORT_ORIGINS.get(name)
        if cached is not None and cached[0] == filename:
            kind, value = cached[1:]
            if kind == 'local':
                local[name] = value
            elif kind == 'dependency':
                dependencies.add(value)
            continue
        path = Path(filename).resolve()
        # PyTorch exposes these virtual module labels, not filesystem paths.
        if name in ('torch.ops', 'torch.classes') and not path.is_file():
            continue
        if inside(path):
            local[name] = str(path.relative_to(ROOT))
            _IMPORT_ORIGINS[name] = (filename, 'local', local[name])
        elif 'site-packages' in path.parts or 'dist-packages' in path.parts:
            dependencies.add(str(path))
            _IMPORT_ORIGINS[name] = (filename, 'dependency', str(path))
        elif path.is_relative_to(stdlib):
            _IMPORT_ORIGINS[name] = (filename, 'stdlib', '')
            continue
        else:
            invalid[name] = str(path)
    if invalid:
        raise RuntimeError('Imports escaped repository/dependency installations: ' + json.dumps(invalid))
    return {'local_modules': local, 'dependency_file_count': len(dependencies), 'invalid': invalid}


def manifest_rows():
    with (ROOT / 'manifests/CHECKPOINTS.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def checked_row(checkpoint):
    path = Path(checkpoint).resolve()
    if not inside(path, ROOT / 'checkpoints'):
        raise ValueError('Checkpoint must be a local known file under repository checkpoints/')
    matches = [r for r in manifest_rows() if (ROOT / r['repo_relative']).resolve() == path]
    if len(matches) != 1:
        raise ValueError('Checkpoint is absent or ambiguous in CHECKPOINTS.csv')
    row = matches[0]
    expected = ROOT / row['repo_relative']
    if any(p.is_symlink() for p in (expected, *expected.parents) if inside(p)):
        raise ValueError('Migrated checkpoint path must contain ordinary files/directories, without symlinks')
    if sha256(path) != row['dest_sha256']:
        raise ValueError('Checkpoint SHA-256 differs from CHECKPOINTS.csv; refusing deserialization')
    return row


def load_known(path):
    """Restricted unpickler, including the NumPy RNG values in known controls."""
    import numpy as np
    import torch
    allowed = [np.core.multiarray._reconstruct, np.core.multiarray.scalar, np.ndarray, np.dtype]
    allowed += [type(np.dtype(t)) for t in ('float32', 'float64', 'uint32', 'int64')]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def state_dict(payload):
    import torch
    for key in ('state_dict', 'model_state_dict'):
        if key in payload:
            return payload[key]
    if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    # Read only the DLinear arm of a hash-verified original container in audit.
    if 'models' in payload and 'dlinear' in payload['models']:
        return payload['models']['dlinear']['model_state_dict']
    raise ValueError('Unrecognized state dictionary schema')


def metadata_for(row, payload, metadata_path=None):
    meta = {}
    metadata_path = Path(metadata_path or ROOT / 'manifests/INFERENCE_METADATA.json')
    if metadata_path.is_file():
        public = json.loads(metadata_path.read_text())
        catalog = public.get('checkpoints', public.get('entries', public))
        if isinstance(catalog, list):
            catalog = {v['repo_relative']: v for v in catalog}
        meta.update(catalog.get(row['repo_relative'], {}))
        if meta.get('checkpoint_sha256', row['dest_sha256']) != row['dest_sha256']:
            raise ValueError('Public inference metadata is bound to a different checkpoint hash')
    # Fields inside the checkpoint take precedence over the public sidecar.
    for obj in (payload.get('metadata', {}), payload.get('fold_protocol', {}), payload):
        if isinstance(obj, dict):
            meta.update({k: v for k, v in obj.items() if k not in ('state_dict', 'model_state_dict', 'optimizer_state_dict', 'models') and v is not None})
    fold = meta.get('fold', {})
    if 'scaler_mean' in fold:
        meta['source_scaler'] = {'mean': fold['scaler_mean'], 'std': fold['scaler_std']}
    if 'source_scaler_mean' in meta:
        meta['source_scaler'] = {'mean': meta['source_scaler_mean'], 'std': meta['source_scaler_std']}
    if not isinstance(meta.get('source_scaler'), dict):
        raise ValueError('Missing public source_scaler mean/std; source-server lookup is prohibited')
    import numpy as np
    import torch
    scaler = {}
    for key in ('mean', 'std'):
        value = meta['source_scaler'][key]
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        scaler[key] = np.asarray(value, dtype=np.float32)
        if not np.isfinite(scaler[key]).all() or (key == 'std' and (scaler[key] <= 0).any()):
            raise ValueError('Invalid source scaler')
    meta['source_scaler'] = scaler
    meta['horizon'] = int(meta.get('horizon', meta.get('horizon_hours', row['horizon'])))
    if meta['horizon'] != int(row['horizon']):
        raise ValueError('Checkpoint horizon differs from manifest')
    context = int(meta.get('context', meta.get('context_hours', 168)))
    if context != 168 or meta['horizon'] not in (24, 48):
        raise ValueError('Unsupported public context/horizon; expected context=168 and horizon=24/48')
    meta['seed'] = int(meta.get('seed', row['seed']))
    if meta['seed'] != int(row['seed']):
        raise ValueError('Checkpoint seed differs from manifest')
    return meta


def module(name):
    result = importlib.import_module(name)
    verify_imports()
    return result


def build_model(row, payload, meta, sites):
    """Use existing model constructors. No fit/train/run entry point is called."""
    group, variant = row['group'], row['variant_id']
    horizon = meta['horizon']
    common = dict(channels=sites, seq_len=168, pred_len=horizon, period_len=24,
                  latent_dim=32, phase_layers=2, routers=1, heads=8, dropout=.1, pos_embed=True)
    threshold = meta.get('source_relation_threshold')
    # Payload arm/variant strings are historical serialization identifiers.
    # Keep reading them without exposing provenance paths as model names.
    arm = str(payload.get('arm', meta.get('arm', variant)))
    mode = 'joint'
    if group == 'design_controls':
        if threshold is None:
            raise ValueError('Missing source_relation_threshold for design control')
        model = module('experiments.design_controls.model_variants').build_model(
            payload.get('variant', arm), sites, float(threshold), seed=meta['seed'])
    elif group == 'charged_dlinear':
        model = module('scprc.training.run_charged_linear_loco')._DLinear2to1()
        mode = 'two_variable'
    elif group == 'charged_patchtst' or (group == 'external_baselines' and variant == 'patchtst'):
        model = module('scprc.training.run_charged_patchtst_volume_only_loco').VolumeOnlyPatchTST()
        mode = 'independent'
    elif group == 'charged_timemixer' or (group == 'external_baselines' and variant == 'timemixer'):
        model = module('scprc.training.run_charged_timemixer_matched').MatchedVolumeTimeMixer()
        mode = 'independent'
    elif group == 'external_baselines' and variant == 'dlinear':
        model = module('scprc.models.linear').DLinear(168, horizon, channels=1, individual=False, moving_avg=25)
        mode = 'independent'
    elif group == 'charged_nbeats_graph' or (group == 'external_baselines' and variant == 'nbeats_graph'):
        model = module('scprc.training.nbeats_graph').SharedStationNBEATSBehaviorGraphResidual(
            hidden=128, blocks=4, graph_rank=16, temperature=.15, residual_bound=.20)
    elif group == 'relation_operators':
        if variant == 'deepsets':
            cls = module('scprc.models.phaseformer_deepsets_adapter').PhaseFormerDeepSetsAdapter
        else:
            cls = module('scprc.models.phasewise_attention_set_adapter').PhaseWiseAttentionPhaseFormer
        model = cls(**common, residual_bound=.2)
    elif group in ('phaseformer_h24', 'phaseformer_h48') or payload.get('model_name') == 'phaseformer_exact' or arm in ('exact', 'exact_pcgrad'):
        # This is the same constructor/network wrapper used by the original
        # VariableSitePhaseFormer, including its exact network.* state keys.
        model = module('scprc.baselines.phaseformer_adapter').PhaseFormerUrbanEV(**{k: v for k, v in common.items()})
    elif payload.get('model_name') in ('phase_relation', 'phase_agnostic_daily'):
        model = module('scprc.training.phase_relations').InputPhaseRelationPhaseFormer(
            **common, relation_mode=payload['model_name'] if payload['model_name'] == 'phase_agnostic_daily' else 'phase',
            relation_hidden=16, relation_rank=8)
    elif group in ('scprc_h24_and_controls', 'scprc_mean_gradient', 'independent_gates', 'scprc_h48_and_control', 'gate_removal'):
        flags = meta.get('gate_flags', {})
        if threshold is None:
            if flags.get('use_source_disagreement_gate') is False:
                threshold = 1.0  # Original constructor's unused value for disabled gate.
            else:
                raise ValueError('Missing source threshold for enabled source gate')
        model = module('scprc.models.scprc').SCPRC(
            **common, relation_mode=meta.get('relation_mode', 'phase'), relation_hidden=16, relation_rank=8,
            source_relation_threshold=float(threshold), support_scale=float(meta.get('support_scale', 1.0)), **flags)
    else:
        raise ValueError('Unsupported checkpoint group/schema: ' + group)
    model = model.float().eval()
    model.load_state_dict(state_dict(payload), strict=True)
    verify_imports()
    return model, mode


def predict(model, mode, raw, meta, device='cpu', batch_size=16, duration=None):
    import numpy as np
    import torch
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 168 or min(raw.shape) < 1 or not np.isfinite(raw).all():
        raise ValueError('Input must be finite numeric raw history [B,168,N]')
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    b, _, n = raw.shape
    mean, std = (meta['source_scaler'][k] for k in ('mean', 'std'))
    if mode == 'two_variable':
        if duration is None:
            raise ValueError('CHARGED DLinear uses a learned volume/duration readout: --duration-input is required')
        duration = np.asarray(duration, dtype=np.float32)
        if duration.shape != raw.shape or not np.isfinite(duration).all() or mean.size != 2 or std.size != 2:
            raise ValueError('Duration must match volume [B,168,N]; scaler must contain volume,duration')
        x = np.stack((raw.transpose(0, 2, 1).reshape(-1, 168), duration.transpose(0, 2, 1).reshape(-1, 168)), axis=-1)
        x = (x - mean.reshape(1, 1, 2)) / std.reshape(1, 1, 2)
        out_mean, out_std = mean.flat[0], std.flat[0]
    else:
        if mean.size != 1 or std.size != 1:
            raise ValueError('Volume-only architecture requires scalar source mean/std')
        x = (raw - mean.item()) / std.item()
        if mode == 'independent':
            x = x.transpose(0, 2, 1).reshape(-1, 168, 1)
        out_mean, out_std = mean.item(), std.item()
    model.to(device).eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            result = model(torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device))
            pieces.append(result.detach().cpu().numpy())
    scaled = np.concatenate(pieces)
    if mode != 'joint':
        scaled = scaled.reshape(b, n, meta['horizon']).transpose(0, 2, 1)
    if scaled.shape != (b, meta['horizon'], n) or not np.isfinite(scaled).all():
        raise RuntimeError('Invalid forward shape or nonfinite prediction: ' + str(scaled.shape))
    output = (scaled * out_std + out_mean).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError('Nonfinite inverse-scaled output')
    return output


def external_file(path, output=False):
    path = Path(path).resolve()
    if inside(path):
        raise ValueError('Input/output arrays must be outside the repository')
    if path.suffix != '.npy':
        raise ValueError('Input/output must use .npy format')
    if output and path.exists():
        raise ValueError('Output already exists; choose a new external path')
    return path


def main():
    args = parser().parse_args()
    configure_imports()
    import numpy as np
    import torch
    torch.set_num_threads(max(1, args.threads))
    if args.clip_min is not None and args.clip_max is not None and args.clip_min > args.clip_max:
        raise ValueError('clip-min exceeds clip-max')
    input_path, output_path = external_file(args.input), external_file(args.output, output=True)
    duration = np.load(external_file(args.duration_input), allow_pickle=False) if args.duration_input else None
    row = checked_row(args.checkpoint)
    payload = load_known(ROOT / row['repo_relative'])
    meta = metadata_for(row, payload, args.metadata)
    raw = np.load(input_path, allow_pickle=False)
    if raw.ndim != 3:
        raise ValueError('Expected raw [B,168,N]')
    model, mode = build_model(row, payload, meta, raw.shape[2])
    output = predict(model, mode, raw, meta, args.device, args.batch_size, duration)
    if args.clip_min is not None or args.clip_max is not None:
        output = np.clip(output, args.clip_min, args.clip_max)
    origins = verify_imports()
    with output_path.open('xb') as f:
        np.save(f, output, allow_pickle=False)
    print(json.dumps({'status': 'passed', 'checkpoint': row['repo_relative'], 'group': row['group'],
                      'model_id': row['model_id'], 'variant_id': row['variant_id'], 'dataset_id': row['dataset_id'],
                      'shape': list(output.shape), 'units': 'raw', 'clipping': [args.clip_min, args.clip_max],
                      'output': str(output_path), 'local_import_count': len(origins['local_modules'])}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Synthetic-only runtime acceptance; independent processes, no dataset or training.

By default checks every manifest checkpoint. --representative checks one per
architecture/arm. Optional --source-root reads ONLY hash-verified original
checkpoints and source text to establish migration parity; never imports them.
Only a summary JSON is written. Synthetic arrays remain in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import infer

ROOT = infer.ROOT
REPORT = ROOT / 'manifests/RUNTIME_AUDIT.json'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--representative', action='store_true', help='One checkpoint per distinct architecture/arm, instead of all checkpoints')
    p.add_argument('--source-root', type=Path, help='Read-only original checkpoint/source text root for parity; never added to sys.path')
    p.add_argument('--group', action='append', help='Restrict to a manifest group; repeatable')
    p.add_argument('--retry-failed', action='store_true', help='Rerun failed architectures and merge into the existing completed report')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--timeout', type=int, default=60, help='Seconds per architecture subprocess')
    p.add_argument('--metadata', type=Path, default=ROOT / 'manifests/INFERENCE_METADATA.json')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    return p


def architecture(row):
    return ':'.join(row[k] for k in ('group', 'variant_id', 'dataset_id', 'horizon'))


def runtime_binding(metadata):
    """Bind this execution to current renamed manifests and bundled code."""
    code = sorted({p for folder in ('scripts', 'scprc', 'experiments')
                   for p in (ROOT / folder).rglob('*.py')})
    return {'checkpoint_manifest_sha256': infer.sha256(ROOT / 'manifests/CHECKPOINTS.csv'),
            'inference_metadata_sha256': infer.sha256(metadata),
            'python_sources': {p.relative_to(ROOT).as_posix(): infer.sha256(p) for p in code}}


def install_guards():
    """Fail if an inference path attempts dataset access, networking or fitting."""
    def audit(event, args):
        if event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system'):
            raise RuntimeError('Runtime audit prohibits networking/process launch: ' + event)
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in ('.npy', '.npz', '.zip', '.csv', '.parquet', '.h5'):
                if path != ROOT / 'manifests/CHECKPOINTS.csv':
                    raise RuntimeError('Runtime audit prohibits dataset/array access: ' + str(path))
            mode = args[1]
            if isinstance(mode, str) and any(c in mode for c in 'wax+'):
                raise RuntimeError('Worker prohibits writing files: ' + str(path))
    sys.addaudithook(audit)
    def profile(frame, event, arg):
        if event != 'call':
            return
        name = frame.f_code.co_name
        if name.startswith(('_train', 'train_', 'fit_')) or name in ('run', 'main', 'run_fold', 'trainrun', 'fit', 'train', 'train_arm', '_ordinary_gradient_train'):
            path = Path(frame.f_code.co_filename)
            if infer.inside(path):
                raise RuntimeError('Training entry point called: ' + name)
    sys.setprofile(profile)


def synthetic(sites):
    import numpy as np
    # Fixed, nonconstant and heterogeneous raw history; no observations used.
    t = np.arange(168, dtype=np.float32)[None, :, None]
    n = np.arange(sites, dtype=np.float32)[None, None, :]
    b = np.arange(2, dtype=np.float32)[:, None, None]
    raw = (12 + 3 * np.sin(t / 24 * 2 * np.pi + n * .13) + .02 * t + n * .4 + b).astype(np.float32)
    duration = (1 + .1 * np.cos(t / 12 + n * .07) + b * .05).astype(np.float32)
    return raw, duration


def source_code_evidence(source_root, origins):
    files, different = [], []
    for relative in sorted(set(origins['local_modules'].values())):
        if relative.startswith('scripts/') or not relative.endswith('.py'):
            continue
        old, new = source_root / relative, ROOT / relative
        if not old.is_file():
            different.append({'path': relative, 'reason': 'original source missing'})
        elif infer.sha256(old) == infer.sha256(new):
            files.append(relative)
        else:
            different.append({'path': relative, 'reason': 'source text changed; inspect code migration audit'})
    return {'identical_source_files': files, 'different_source_files': different}


def source_reference(row, payload, meta, sites):
    """Exercise the source runners' own builders independently of CLI routing."""
    group = row['group']
    arm = payload.get('arm')
    threshold = meta.get('source_relation_threshold')
    if group == 'scprc_h24_and_controls':
        name = 'experiments.forecasting.h24_framework.run_canonical_anchor_fold'
        model = infer.module(name).build_model(arm, sites, float(threshold or 1.0))
    elif group in ('phaseformer_h24', 'phaseformer_h48', 'scprc_h48_and_control'):
        protocol = infer.module('scprc.training.training_protocol')
        protocol.HORIZON = meta['horizon']
        if group == 'scprc_h48_and_control' and arm == 'scpcr_candidate':
            name = 'scprc.training.scprc_training_support'
            # Reuse the canonical runner's original constants-only legacy shim
            # before importing v21; no historical training module is needed.
            infer.module('experiments.forecasting.h24_framework.run_canonical_anchor_fold')
            original = infer.module(name)
            original.HORIZON = meta['horizon']
            model = original._build_model(sites, float(threshold), support_scale=1.0)
        else:
            name = 'scprc.training.training_protocol._build_model'
            model = protocol._build_model('phaseformer_exact', sites)
    elif group == 'independent_gates':
        name = 'experiments.independent_gates.run_canonical_component_ablation'
        model = infer.module(name).build_model(arm, sites, float(threshold or 1.0))
    elif group == 'gate_removal':
        name = 'experiments.energy_value.gate_removal.run_strict_component_ablation'
        model = infer.module(name).build_model(arm, sites, float(threshold or 1.0))
    elif group == 'scprc_mean_gradient':
        name = 'scprc.training.scprc_training_support._build_model'
        infer.module('experiments.forecasting.h24_framework.run_canonical_anchor_fold')
        model = infer.module('scprc.training.scprc_training_support')._build_model(sites, float(threshold), support_scale=1.0)
    else:
        model, mode = infer.build_model(row, payload, meta, sites)
        return model, mode, 'bundled original model constructor'
    model = model.float().eval()
    model.load_state_dict(infer.state_dict(payload), strict=True)
    return model, 'joint', name


def worker(args):
    import numpy as np
    import torch
    infer.configure_imports()
    torch.set_num_threads(max(1, args.threads))
    install_guards()
    rows = [r for r in infer.manifest_rows() if architecture(r) == args.worker]
    if args.representative:
        rows = rows[:1]
    results = []
    for row in rows:
        started = time.monotonic()
        result = {'checkpoint': row['repo_relative'], 'group': row['group'], 'architecture': args.worker,
                  **{k: row[k] for k in ('model_id', 'variant_id', 'dataset_id', 'city', 'seed')},
                  'checkpoint_sha256': row['dest_sha256']}
        try:
            infer.checked_row(ROOT / row['repo_relative'])
            payload = infer.load_known(ROOT / row['repo_relative'])
            meta = infer.metadata_for(row, payload, args.metadata)
            # Small fixed synthetic site set tests architecture without depending
            # on a city's data. Record declared site count separately.
            sites = 7
            raw, duration = synthetic(sites)
            model, mode = infer.build_model(row, payload, meta, sites)
            prediction = infer.predict(model, mode, raw, meta, args.device, 8,
                                       duration if mode == 'two_variable' else None)
            if mode == 'joint':
                # Independent transcription of the source protocol's raw ->
                # standardized -> forward -> raw sequence (no infer.predict).
                mean = float(meta['source_scaler']['mean'].item())
                std = float(meta['source_scaler']['std'].item())
                with torch.inference_mode():
                    direct = model(torch.from_numpy((raw - mean) / std).to(args.device)).cpu().numpy()
                direct_raw = (direct * std + mean).astype(np.float32)
                if not np.array_equal(direct_raw, prediction):
                    raise AssertionError('Raw source scaling/inverse differs from direct source protocol')
                result['raw_source_scaling'] = {'status': 'passed', 'max_abs_difference': 0.0,
                                                'input': '(raw-mean)/std', 'output': 'standardized_prediction*std+mean',
                                                'default_clipping': False}
            if any(p.grad is not None for p in model.parameters()) or model.training:
                raise AssertionError('Model left inference mode or created gradients')
            result.update(status='passed', strict_state_load=True, finite=True,
                          shape=list(prediction.shape), declared_site_count=meta.get('site_count'),
                          synthetic_site_count=sites, horizon=meta['horizon'], mode=mode,
                          state_tensor_count=len(infer.state_dict(payload)))
            if args.source_root:
                # Original payload is known only after validating manifest hash.
                relative = Path(row['original_source'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Original checkpoint must have a lexical path within explicit source root')
                source_path = args.source_root.resolve() / relative
                resolved = source_path.resolve()
                # Original export symlinks may be relocated by the source owner.
                # Trust the known manifest bytes, never a path alone; retain the
                # recorded resolved_source verbatim as historical provenance.
                if (source_path.stat().st_size != int(row['source_bytes'])
                        or infer.sha256(source_path) != row['source_sha256']):
                    raise ValueError('Original checkpoint SHA-256 differs from manifest')
                original = infer.load_known(source_path)
                old_state, new_state = infer.state_dict(original), infer.state_dict(payload)
                equal = old_state.keys() == new_state.keys() and all(torch.equal(old_state[k], new_state[k]) for k in old_state)
                if not equal:
                    raise AssertionError('Original/migrated state tensor mismatch')
                reference, reference_mode, reference_constructor = source_reference(row, original, meta, sites)
                reference_prediction = infer.predict(reference, reference_mode, raw, meta, args.device, 8,
                                                    duration if reference_mode == 'two_variable' else None)
                maximum = float(np.max(np.abs(reference_prediction - prediction)))
                if not np.array_equal(reference_prediction, prediction):
                    raise AssertionError('Original/migrated forward mismatch: ' + str(maximum))
                result['parity'] = {'status': 'passed', 'state_tensors_exact': True,
                                    'raw_output_exact': True, 'max_abs_difference': maximum,
                                    'reference_constructor': reference_constructor,
                                    'source_sha256': row['source_sha256'],
                                    'source_resolved_path': str(resolved),
                                    'recorded_resolved_source': row['resolved_source'],
                                    'method': 'original and migrated weights, independent models, bundled source constructors; original source is read/hashed and never imported'}
            origins = infer.verify_imports()
            result['import_origins_passed'] = True
        except Exception as error:
            result.update(status='failed', error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(limit=5))
        result['seconds'] = round(time.monotonic() - started, 3)
        results.append(result)
    origins = infer.verify_imports()
    record = {'architecture': args.worker, 'results': results, 'imports': origins,
              'cwd': str(Path.cwd()), 'dataset_reads': 0, 'array_files_written': 0,
              'training_calls': 0, 'network_access': 0}
    if args.source_root:
        record['source_code_evidence'] = source_code_evidence(args.source_root.resolve(), origins)
    sys.setprofile(None)
    print('RUNTIME_RESULT=' + json.dumps(record, default=str, allow_nan=False))


def save_report(report):
    results = [r for worker in report['workers'] for r in worker['results']]
    report['counts'] = dict(Counter(r['status'] for r in results))
    groups = defaultdict(Counter)
    for row in results:
        groups[row['group']][row['status']] += 1
    report['groups'] = {g: dict(c) for g, c in groups.items()}
    report['tested_checkpoints'] = len(results)
    counts = Counter(r['checkpoint'] for r in results)
    expected = set(report['expected_checkpoint_paths'])
    report['coverage'] = {'expected': len(expected), 'unique_tested': len(counts),
                          'missing': sorted(expected - counts.keys()),
                          'unexpected': sorted(counts.keys() - expected),
                          'duplicates': sorted(k for k, v in counts.items() if v != 1)}
    complete = (set(counts) == expected and all(v == 1 for v in counts.values()))
    report['coverage']['complete'] = complete
    report['status'] = 'running' if report['remaining_architectures'] else ('passed' if complete and results and all(r['status'] == 'passed' for r in results) and report['binding_matches_current'] else 'failed')
    REPORT.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')


def main():
    args = parser().parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run with cwd equal to repository root: ' + str(ROOT))
    if args.worker:
        worker(args)
        return
    rows = infer.manifest_rows()
    if args.group:
        rows = [r for r in rows if r['group'] in args.group]
    keys = sorted(set(map(architecture, rows)))
    if not keys:
        raise ValueError('No matching checkpoints')
    expected = [r['repo_relative'] for r in rows]
    if args.representative:
        expected = [next(r['repo_relative'] for r in rows if architecture(r) == key) for key in keys]
    report = {'schema_version': 2, 'device': args.device, 'cwd': str(ROOT),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'binding': runtime_binding(args.metadata), 'binding_matches_current': True,
              'expected_checkpoint_paths': expected,
              'selection': 'representative' if args.representative else 'all',
              'manifest_checkpoint_count': len(rows), 'architecture_count': len(keys),
              'metadata': str(args.metadata.relative_to(ROOT)) if infer.inside(args.metadata) else str(args.metadata),
              'source_parity_requested': bool(args.source_root),
              'synthetic': {'batch': 2, 'context': 168, 'sites': 7, 'seed': 'fixed analytic sin/cos; no RNG/data'},
              'limitations': ['Synthetic checks establish execution and migration consistency, not forecast quality.',
                             'Source parity uses original hash-verified weights with bundled constructors; source-server modules are never imported.',
                             'CHARGED DLinear requires raw duration as well as volume.'],
              'workers': [], 'remaining_architectures': keys.copy()}
    if args.retry_failed:
        existing = json.loads(REPORT.read_text())
        if existing.get('remaining_architectures') or existing['device'] != args.device:
            raise ValueError('Retry requires a completed report with the same device')
        if (existing.get('binding') != report['binding']
                or existing.get('selection') != report['selection']
                or existing.get('expected_checkpoint_paths') != expected
                or existing.get('source_parity_requested') != bool(args.source_root)):
            raise ValueError('Retry binding/coverage changed; run a fresh acceptance audit')
        failed = {w['architecture'] for w in existing['workers'] if any(r['status'] != 'passed' for r in w['results'])}
        keys = [k for k in keys if k in failed]
        if not keys:
            raise ValueError('No failed architectures selected for retry')
        report = existing
        report.setdefault('retry_history', []).append({'architectures': keys.copy(), 'previous_counts': report['counts'].copy(),
                                                      'reason': 'rerun failed architectures with identical code/manifest binding'})
        report['workers'] = [w for w in report['workers'] if w['architecture'] not in keys]
        report['remaining_architectures'] = keys.copy()
    save_report(report)
    for key in keys:
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', key,
               '--device', args.device, '--threads', str(args.threads), '--metadata', str(args.metadata)]
        if args.representative:
            cmd.append('--representative')
        if args.source_root:
            cmd += ['--source-root', str(args.source_root.resolve())]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='')
        try:
            process = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
            marker = next((s for s in reversed(process.stdout.splitlines()) if s.startswith('RUNTIME_RESULT=')), None)
            if not marker:
                raise RuntimeError((process.stderr + process.stdout)[-5000:])
            record = json.loads(marker.split('=', 1)[1])
        except Exception as error:
            selected = [r for r in rows if architecture(r) == key]
            if args.representative:
                selected = selected[:1]
            record = {'architecture': key, 'results': [{'checkpoint': r['repo_relative'], 'group': r['group'], 'architecture': key,
                      'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)} for r in selected]}
        report['workers'].append(record)
        report['remaining_architectures'].remove(key)
        save_report(report)
        print(json.dumps({'architecture': key, 'counts': dict(Counter(r['status'] for r in record['results']))}), flush=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['binding_matches_current'] = report['binding'] == runtime_binding(args.metadata)
    save_report(report)
    print(json.dumps({'report': str(REPORT.relative_to(ROOT)), 'status': report['status'], 'counts': report['counts']}))
    if report['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()

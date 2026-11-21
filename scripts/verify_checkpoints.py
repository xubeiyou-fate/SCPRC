#!/usr/bin/env python3
"""Verify checkpoint manifests, local ZIPs, and optional trusted source parity.

Default: python scripts/verify_checkpoints.py
Full local source audit: python scripts/verify_checkpoints.py --source-root /data/TimeMixer
No training, network access, or source writes. torch.load(weights_only=False) is
used only with --source-root on these explicitly trusted local research products.
ZIP and destination checks use SHA256 and do not deserialize checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

CHECKPOINT_FIELDS = (
    'group role original_source resolved_source repo_relative source_bytes '
    'dest_bytes source_sha256 dest_sha256 migration_mode payload_format '
    'retained_fields omitted_fields tensor_parity forward_parity release_asset '
    'model_id variant_id city seed dataset_id horizon'
).split()
MAX_ASSET_BYTES = 2 * 1024 ** 3
DLINEAR_GROUP = 'charged_dlinear'
DLINEAR_SOURCE = 'research/urbanev_timemixer/experiments/run_charged_linear_loco.py'
DLINEAR_LEGACY_GROUP = 'charged_dlinear_two_variable_containers'
DLINEAR_OMITTED = [
    'models.nlinear', 'models.dlinear.optimizer_state_dict',
    'models.dlinear.torch_rng_state', 'models.dlinear.numpy_rng_state',
]


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_repo_path(root, relative, prefix=None):
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or '..' in path.parts
            or '\\' in relative or str(path) != relative):
        raise ValueError(f'Unsafe repository path: {relative}')
    if prefix and (not path.parts or path.parts[0] != prefix):
        raise ValueError(f'Expected {prefix}/ path: {relative}')
    target = Path(root).joinpath(*path.parts)
    # Reject symlink ancestors even when they currently resolve inside root.
    current = Path(root)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink is not allowed: {relative}')
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Path escapes repository: {relative}')
    return target


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def checkpoint_rows(root):
    """Read renamed paths/semantics without rewriting source provenance."""
    rows = csv_rows(Path(root) / 'manifests/CHECKPOINTS.csv')
    if not rows or not set(CHECKPOINT_FIELDS).issubset(rows[0]):
        raise ValueError('Checkpoint manifest missing or incompatible schema')
    seen = set()
    for row in rows:
        key = row['repo_relative']
        safe_repo_path(root, key, 'checkpoints')
        if key in seen:
            raise ValueError(f'Duplicate checkpoint path: {key}')
        seen.add(key)
        if any(not row[k] for k in ('group', 'model_id', 'variant_id', 'seed', 'dataset_id', 'horizon')):
            raise ValueError(f'Missing checkpoint semantics: {key}')
        if row['dataset_id'] == 'CHARGED' and not row['city']:
            raise ValueError(f'Missing CHARGED target city: {key}')
        if int(row['horizon']) not in (24, 48):
            raise ValueError(f'Unsupported checkpoint horizon: {key}')
        int(row['seed'])
        for field in ('source_sha256', 'dest_sha256'):
            digest = row[field]
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Invalid {field}: {key}')
    index_path = Path(root) / 'manifests/CHECKPOINT_SHA256.csv'
    if index_path.is_file():
        index = csv_rows(index_path)
        expected = {row['repo_relative']: (row['dest_sha256'], int(row['dest_bytes'])) for row in rows}
        if any(set(row) != {'repo_relative', 'sha256', 'bytes'} for row in index):
            raise ValueError('Incompatible CHECKPOINT_SHA256.csv schema')
        actual = {row['repo_relative']: (row['sha256'], int(row['bytes'])) for row in index}
        if len(actual) != len(index) or actual != expected:
            raise ValueError('CHECKPOINT_SHA256.csv differs from CHECKPOINTS.csv')
    return rows


def model_state(payload):
    import torch
    if isinstance(payload, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in payload:
                return payload[key]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError('Unrecognized model state format')


def tensor_parity(left, right):
    """Compare shape, dtype, keys and every byte, including NaN bit patterns."""
    import torch
    if set(left) != set(right):
        raise ValueError('Model state keys differ')
    details = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError(f'Non-tensor model entry: {key}')
        aa, bb = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        if (aa.shape != bb.shape or aa.dtype != bb.dtype
                or not torch.equal(aa.reshape(-1).view(torch.uint8),
                                   bb.reshape(-1).view(torch.uint8))):
            raise ValueError(f'Tensor bytes differ: {key}')
        digest = hashlib.sha256(aa.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        details.append({'key': key, 'shape': list(aa.shape), 'dtype': str(aa.dtype),
                        'numel': aa.numel(), 'sha256': digest, 'bitwise_equal': True})
    return details


def trusted_source_dlinear(source_root):
    """Evaluate only the two model class ASTs, never the training runner imports."""
    import torch
    source_path = Path(source_root) / DLINEAR_SOURCE
    module = ast.parse(source_path.read_text(encoding='utf-8'))
    names = {'_MovingAverage', '_DLinear2to1'}
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(classes) != 2:
        raise ValueError('Original DLinear classes not found')
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'CONTEXT', 'HORIZON', 'MOVING_AVG'}:
                    constants[target.id] = ast.literal_eval(node.value)
    adapter_path = Path(source_root) / 'research/urbanev_timemixer/experiments/run_charged_covariate_hptn_phaseformer_loco.py'
    for node in ast.parse(adapter_path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VARIABLES' for t in node.targets):
            constants['VARIABLES'] = ast.literal_eval(node.value)
    if constants != {'CONTEXT': 168, 'HORIZON': 24, 'MOVING_AVG': 25, 'VARIABLES': ('volume', 'duration')}:
        raise ValueError('Unexpected original DLinear configuration')
    namespace = {'torch': torch, 'nn': torch.nn, **constants}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace['_DLinear2to1']


def dlinear_forward_parity(source_state, export_state, source_root):
    import torch
    model_class = trusted_source_dlinear(source_root)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        source_model, export_model = model_class().eval(), model_class().eval()
    source_model.load_state_dict(source_state, strict=True)
    export_model.load_state_dict(export_state, strict=True)
    synthetic = torch.linspace(-1.0, 1.0, 168 * 2, dtype=torch.float32).reshape(1, 168, 2)
    with torch.inference_mode():
        a, b = source_model(synthetic), export_model(synthetic)
    equal = torch.equal(a, b) and bool(torch.isfinite(a).all())
    if not equal:
        raise ValueError('DLinear synthetic forward parity failed')
    return {'status': 'passed', 'method': 'original model class AST; strict=True; CPU eval inference_mode',
            'input': 'synthetic torch.linspace(-1, 1, 336); no dataset access',
            'input_shape': list(synthetic.shape), 'output_shape': list(a.shape),
            'input_sha256': hashlib.sha256(synthetic.numpy().tobytes()).hexdigest(),
            'bitwise_equal': True, 'max_abs_error': float((a - b).abs().max()),
            'model_source': DLINEAR_SOURCE,
            'model_source_sha256': sha256_file(Path(source_root) / DLINEAR_SOURCE)}


def inspect_payload(payload):
    """Classify EVERY array/list leaf; fail closed on unrecognized arrays.

    Recognized non-data arrays are model parameters/buffers, optimizer moments,
    PRNG states and small source-only scalers. Strings and short scalar config
    lists (e.g. evaluation interval endpoints) are metadata, never raw labels.
    """
    import numpy as np
    import torch
    state = model_state(payload)
    tensor_parity(state, state)
    arrays, sequences, unclassified, data_fields = [], [], [], []
    data_names = {'truth', 'raw_truth', 'truth_raw', 'truth_volume_raw', 'input', 'inputs',
                  'raw_input', 'raw_inputs', 'x', 'y', 'y_true', 'labels', 'raw_labels',
                  'prediction', 'predictions', 'prediction_raw', 'prediction_dlinear_raw',
                  'prediction_nlinear_raw', 'features', 'data_array', 'train_data',
                  'val_data', 'test_data', 'raw_data', 'timeseries'}

    def walk(value, path='', context='metadata'):
        if value is state:
            arrays.append({'path': path or '<root>', 'classification': 'model_parameters_and_buffers',
                           'tensor_count': len(state)})
            return
        if isinstance(value, dict):
            for key, item in value.items():
                key = str(key)
                child = f'{path}.{key}' if path else key
                ctx = context
                if key == 'optimizer_state_dict':
                    ctx = 'optimizer_moments'
                elif 'rng_state' in key:
                    ctx = 'prng_state'
                elif key.startswith('scaler_') or key.startswith('source_scaler'):
                    ctx = 'source_only_scaler'
                if key.lower() in data_names:
                    # No raw data or label arrays are allowed, regardless of size.
                    if isinstance(item, (dict, list, tuple, np.ndarray)) or torch.is_tensor(item):
                        data_fields.append(child)
                walk(item, child, ctx)
        elif torch.is_tensor(value) or isinstance(value, np.ndarray):
            record = {'path': path, 'type': type(value).__name__, 'shape': list(value.shape),
                      'dtype': str(value.dtype), 'classification': context}
            arrays.append(record)
            if context not in {'optimizer_moments', 'prng_state', 'source_only_scaler'}:
                unclassified.append(path)
            array_size = value.size if isinstance(value, np.ndarray) else value.numel()
            if context == 'source_only_scaler' and array_size > 32:
                unclassified.append(path)
        elif isinstance(value, (list, tuple)):
            sequences.append({'path': path, 'length': len(value), 'classification': context})
            if len(value) > 32 and context not in {'optimizer_moments', 'prng_state'}:
                unclassified.append(path)
            for item in value:
                walk(item, f'{path}[]', context)
        elif not isinstance(value, (str, int, float, bool, type(None), np.generic)):
            unclassified.append(path)
    walk(payload)
    if unclassified or data_fields:
        raise ValueError(f'Unexpected raw/array fields: {sorted(set(unclassified + data_fields))}')
    return {'raw_data_arrays_found': [], 'arrays': arrays, 'sequences': sequences,
            'model_tensor_count': len(state)}


def verify_assets(root, checkpoint_rows, asset_rows):
    by_path = {r['repo_relative']: r for r in checkpoint_rows}
    seen_paths, stored_paths, aliases_seen = set(), set(), set()
    for asset in asset_rows:
        relative = asset['repo_relative']
        path = safe_repo_path(root, relative, 'release_assets')
        if not path.is_file() or path.stat().st_size != int(asset['bytes']):
            raise ValueError(f'ZIP size/missing: {relative}')
        if path.stat().st_size >= MAX_ASSET_BYTES or sha256_file(path) != asset['sha256']:
            raise ValueError(f'ZIP hash/limit: {relative}')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError(f'Duplicate ZIP names: {relative}')
            for info in archive.infolist():
                safe_repo_path(root, info.filename)
                if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                    raise ValueError('Unexpected symlink/directory in ZIP')
            sums = {}
            for line in archive.read('SHA256SUMS').decode('utf-8').splitlines():
                digest, member = line.split('  ', 1)
                if member in sums:
                    raise ValueError('Duplicate SHA entry')
                sums[member] = digest
            if set(sums) != set(names) - {'SHA256SUMS'}:
                raise ValueError('ZIP SHA manifest coverage mismatch')
            for member, digest in sums.items():
                with archive.open(member) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError(f'ZIP member hash mismatch: {member}')
            packed = list(csv.DictReader(io.StringIO(archive.read('CHECKPOINTS.csv').decode('utf-8'))))
            alias_map = json.loads(archive.read('CHECKPOINT_ALIASES.json'))
            if len(packed) != int(asset['checkpoint_count']) or len(alias_map) != int(asset['alias_count']):
                raise ValueError('ZIP manifest counts differ')
            actual_payloads = set(names) - {'SHA256SUMS', 'CHECKPOINTS.csv', 'CHECKPOINT_ALIASES.json'}
            if len(actual_payloads) != int(asset['stored_checkpoint_count']):
                raise ValueError('ZIP stored count differs')
            for row in packed:
                key = row['repo_relative']
                if key in seen_paths or key not in by_path or row != by_path[key]:
                    raise ValueError(f'ZIP/CSV identity mismatch: {key}')
                if row['release_asset'] != relative or row['group'] != asset['group']:
                    raise ValueError('ZIP group or asset mismatch')
                seen_paths.add(key)
                member = alias_map.get(key, key)
                if member not in actual_payloads or sums[member] != row['dest_sha256']:
                    raise ValueError(f'ZIP checkpoint mapping mismatch: {key}')
                if key in alias_map:
                    if member not in by_path or member in alias_map:
                        raise ValueError('ZIP alias must point directly to canonical checkpoint')
                    aliases_seen.add(key)
                else:
                    if row['dest_sha256'] in stored_paths:
                        raise ValueError('Duplicate weights stored in ZIPs')
                    stored_paths.add(row['dest_sha256'])
            if set(alias_map) != {r['repo_relative'] for r in packed} - actual_payloads:
                raise ValueError('ZIP alias coverage differs')
    if seen_paths != set(by_path):
        raise ValueError('Assets do not cover all checkpoints')
    return {'asset_count': len(asset_rows), 'stored_unique_checkpoints': len(stored_paths),
            'restored_aliases': len(aliases_seen), 'covered_checkpoint_paths': len(seen_paths)}


def verify(root, source_root=None, check_assets=True):
    rows = checkpoint_rows(root)
    expected = set()
    dlinear_count = 0
    total_tensors = 0
    if source_root:
        import torch
        torch.set_num_threads(1)
    for row in rows:
        key = row['repo_relative']
        if key in expected:
            raise ValueError('Unexpected checkpoint path or duplicate')
        expected.add(key)
        path = safe_repo_path(root, key, 'checkpoints')
        if not path.is_file() or path.stat().st_size != int(row['dest_bytes']):
            raise ValueError(f'Checkpoint missing/size mismatch: {key}')
        if sha256_file(path) != row['dest_sha256']:
            raise ValueError(f'Checkpoint SHA256 mismatch: {key}')
        if row['migration_mode'] == 'copy' and row['source_sha256'] != row['dest_sha256']:
            raise ValueError('Copied checkpoint source/destination hashes differ')
        if source_root:
            relative = PurePosixPath(row['original_source'])
            if relative.is_absolute() or '..' in relative.parts or chr(92) in row['original_source']:
                raise ValueError('Original checkpoint must stay lexically within source root')
            # Original export symlinks are allowed only after the manifest
            # size/hash check below, before any checkpoint deserialization.
            source = Path(source_root) / relative
            if source.stat().st_size != int(row['source_bytes']) or sha256_file(source) != row['source_sha256']:
                raise ValueError(f'Source changed: {row["original_source"]}')
            a = torch.load(source, map_location='cpu', weights_only=False)
            b = torch.load(path, map_location='cpu', weights_only=False)
            if row['group'] == DLINEAR_GROUP:
                if set(b) != {'model_state_dict', 'metadata'}:
                    raise ValueError('DLinear export contains unexpected top-level fields')
                md = b['metadata']
                if (md['format'] != 'dlinear-only-v1' or md['source_sha256'] != row['source_sha256']
                        or md['group'] not in (row['group'], DLINEAR_LEGACY_GROUP)
                        or md['original_source'] != row['original_source']):
                    raise ValueError('DLinear metadata mismatch')
                details = tensor_parity(a['models']['dlinear']['model_state_dict'], b['model_state_dict'])
                for k, v in a['fold'].items():
                    vv = md['fold'][k]
                    if k in {'scaler_mean', 'scaler_std'}:
                        tensor_parity({k: torch.from_numpy(v)}, {k: vv})
                    elif v != vv:
                        raise ValueError('DLinear fold metadata changed')
                dlinear_forward_parity(a['models']['dlinear']['model_state_dict'], b['model_state_dict'], source_root)
                dlinear_count += 1
            else:
                details = tensor_parity(model_state(a), model_state(b))
            total_tensors += len(details)
            inspect_payload(b)
    actual = set()
    for path in (root / 'checkpoints').rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Checkpoint tree contains symlink: {path}')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError('Checkpoint tree differs from manifest')
    result = {'status': 'passed', 'checkpoint_count': len(rows),
              'checkpoint_bytes': sum(int(r['dest_bytes']) for r in rows),
              'source_verification': bool(source_root)}
    if source_root:
        result.update({'verified_model_tensors': total_tensors,
                       'dlinear_tensor_and_forward_parity_count': dlinear_count})
    if check_assets:
        result['assets'] = verify_assets(root, rows, csv_rows(root / 'manifests/RELEASE_ASSETS.csv'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path, help='Trusted original local research root')
    parser.add_argument('--skip-assets', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root.resolve(), args.source_root, not args.skip_assets), indent=2))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Checkpoint verification failed: {exc}\n')


if __name__ == '__main__':
    main()

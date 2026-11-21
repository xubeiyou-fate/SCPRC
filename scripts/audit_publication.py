#!/usr/bin/env python3
"""Audit the local release tree without copying datasets or using the network."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.npy', '.npz', '.xlsx', '.xls', '.parquet', '.h5', '.hdf5'}
SECRET_PATTERNS = {
    'github_token': re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    'github_fine_grained_token': re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),
    'aws_access_key': re.compile(r'AKIA[A-Z0-9]{16}'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'manifests/PUBLICATION_TREE_AUDIT.json')
    args = ap.parse_args()
    violations, secret_findings, compile_errors, links = [], [], [], []
    files, total, weights, archives = 0, 0, 0, 0
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in {'.git', '.venv', '__pycache__', '.pytest_cache'} for part in rel.parts):
            continue
        if path.is_symlink():
            links.append(str(rel))
            continue
        if not path.is_file():
            continue
        files += 1
        size = path.stat().st_size
        total += size
        if path.suffix in FORBIDDEN_SUFFIXES:
            violations.append({'path': str(rel), 'reason': 'dataset_or_raw_array_extension'})
        if path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors'}:
            weights += 1
            if rel.parts[0] != 'checkpoints':
                violations.append({'path': str(rel), 'reason': 'weight_outside_checkpoints'})
        if path.suffix == '.zip':
            archives += 1
            if rel.parts[0] != 'release_assets':
                violations.append({'path': str(rel), 'reason': 'archive_outside_release_assets'})
        if size > 100 * 1024 * 1024 and rel.parts[0] not in {'checkpoints', 'release_assets'}:
            violations.append({'path': str(rel), 'reason': 'oversized_normal_git_file'})
        if path.suffix == '.py':
            try:
                compile(path.read_bytes(), str(rel), 'exec')
            except (SyntaxError, UnicodeDecodeError) as exc:
                compile_errors.append({'path': str(rel), 'error': str(exc)})
        if size <= 4 * 1024 * 1024 and path.suffix in {'.py', '.sh', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.csv'}:
            text = path.read_text(encoding='utf-8', errors='replace')
            for name, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(text):
                    secret_findings.append({'path': str(rel), 'rule': name, 'line': text.count('\n', 0, match.start()) + 1})
    result = {
        'date': '2026-11-21', 'files': files, 'total_bytes': total,
        'weight_files': weights, 'release_zip_files': archives,
        'data_or_size_violations': violations, 'secret_pattern_findings': secret_findings,
        'python_compile_errors': compile_errors, 'symlinks': links,
        'scope': 'extension/size/symlink/known-secret-pattern and Python syntax checks; not a full license or scientific audit',
        'passed': not (violations or secret_findings or compile_errors or links),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

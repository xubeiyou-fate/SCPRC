#!/usr/bin/env python3
"""Verify the local migration file manifest; no network or dataset access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-only', action='store_true', help='Verify only ordinary Git files, excluding checkpoints and Release ZIPs')
    args = ap.parse_args()
    manifest = ROOT/'manifests/MANIFEST_SHA256.csv'
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    checked, failures = 0, []
    for row in rows:
        if args.code_only and row['channel'] != 'git':
            continue
        p = ROOT/row['path']
        if not p.resolve().is_relative_to(ROOT) or p.is_symlink():
            failures.append({'path': row['path'], 'reason': 'unsafe_path'})
            continue
        if not p.is_file():
            failures.append({'path': row['path'], 'reason': 'missing'})
            continue
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        checked += 1
        if p.stat().st_size != int(row['bytes']) or h.hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'reason': 'size_or_sha256_mismatch'})
    print(json.dumps({'status': 'passed' if not failures else 'failed', 'checked': checked, 'code_only': args.code_only, 'failures': failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

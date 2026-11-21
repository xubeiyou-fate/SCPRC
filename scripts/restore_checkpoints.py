#!/usr/bin/env python3
"""Restore local release ZIPs to ordinary files, including shared-weight aliases.

  python scripts/restore_checkpoints.py
  python scripts/restore_checkpoints.py --output-root /tmp/restored-checkpoints
  python scripts/restore_checkpoints.py --group scprc_h24_and_controls

Requires manifests/CHECKPOINTS.csv, manifests/RELEASE_ASSETS.csv and the local
release_assets/*.zip. No downloads or torch deserialization. Verifies archive,
member and final-file SHA256. Existing identical files are retained; differing
files cause an error. ZIP metadata is read directly and is never extracted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import verify_checkpoints as verify


def restore(repo_root, output_root, groups=None):
    rows = verify.checkpoint_rows(repo_root)
    assets = verify.csv_rows(repo_root / 'manifests/RELEASE_ASSETS.csv')
    available_groups = {row['group'] for row in rows}
    if groups:
        unknown = set(groups) - available_groups
        if unknown:
            raise ValueError(f'Unknown groups: {sorted(unknown)}')
        rows = [row for row in rows if row['group'] in groups]
        assets = [asset for asset in assets if asset['group'] in groups]
    # Complete archive verification before creating any destination files.
    coverage = verify.verify_assets(repo_root, rows, assets)
    expected = {row['repo_relative']: row for row in rows}
    for key, row in expected.items():
        dest = verify.safe_repo_path(output_root, key, 'checkpoints')
        if dest.exists() and (not dest.is_file() or verify.sha256_file(dest) != row['dest_sha256']):
            raise ValueError(f'Existing destination differs: {key}')
    written = 0
    unchanged = 0

    def write_verified(relative, reader, expected_hash):
        nonlocal written, unchanged
        target = verify.safe_repo_path(output_root, relative, 'checkpoints')
        if target.exists():
            unchanged += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        verify.safe_repo_path(output_root, relative, 'checkpoints')
        with tempfile.NamedTemporaryFile(prefix='.restore-', dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with reader() as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                stream.flush()
                if verify.sha256_file(temporary) != expected_hash:
                    raise ValueError(f'Restored file hash differs: {relative}')
                os.chmod(temporary, 0o644)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        written += 1

    for asset in assets:
        path = verify.safe_repo_path(repo_root, asset['repo_relative'], 'release_assets')
        with zipfile.ZipFile(path) as archive:
            aliases = json.loads(archive.read('CHECKPOINT_ALIASES.json'))
            selected = [row for row in rows if row['release_asset'] == asset['repo_relative']]
            for row in selected:
                key = row['repo_relative']
                if key not in aliases:
                    write_verified(key, lambda key=key: archive.open(key), row['dest_sha256'])
            for row in selected:
                key = row['repo_relative']
                if key in aliases:
                    canonical = verify.safe_repo_path(output_root, aliases[key], 'checkpoints')
                    if verify.sha256_file(canonical) != row['dest_sha256']:
                        raise ValueError('Canonical alias source differs')
                    write_verified(key, lambda canonical=canonical: canonical.open('rb'), row['dest_sha256'])
    for key, row in expected.items():
        target = verify.safe_repo_path(output_root, key, 'checkpoints')
        if target.stat().st_size != int(row['dest_bytes']) or verify.sha256_file(target) != row['dest_sha256']:
            raise ValueError(f'Final restored file mismatch: {key}')
    return {'status': 'passed', 'checkpoint_count': len(rows), 'written': written,
            'already_identical': unchanged, 'bytes': sum(int(row['dest_bytes']) for row in rows),
            'output_root': str(output_root), 'asset_coverage': coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-root', type=Path, help='Destination root; defaults to repository')
    parser.add_argument('--group', action='append', help='Restore only this group; repeatable')
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        print(json.dumps(restore(root, (args.output_root or root).resolve(), args.group), indent=2))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Checkpoint restoration failed: {exc}\n')


if __name__ == '__main__':
    main()

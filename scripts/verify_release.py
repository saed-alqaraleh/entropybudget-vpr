#!/usr/bin/env python3
"""Verify original-source fingerprints, source inventories, and analysis tables."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check-inventory',action='store_true')
    p.add_argument('--check-tables',action='store_true')
    args=p.parse_args()
    if not args.check_inventory and not args.check_tables:
        args.check_inventory=args.check_tables=True
    fingerprints=json.loads((ROOT/'source_fingerprints.json').read_text())
    executed=fingerprints['recorded_executed_sha256']
    released=fingerprints['released_sha256']
    assert len(released)==5 and len(executed)==5
    for name,h in released.items():assert sha(ROOT/name)==h,name
    mapping=fingerprints.get('path_mapping',{})
    matching=sum(executed[name]==released[mapping.get(name,name)] for name in executed)
    assert matching==4==fingerprints['released_files_matching_executed_sha256']
    print('PASS: five protected source fingerprints; four match recorded executed source')
    if args.check_inventory:
        manifest=json.loads((ROOT/'SHA256_FILE_INVENTORY.json').read_text())
        all_paths={str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and '.git' not in f.parts and f.name!='SHA256_FILE_INVENTORY.json' and '__pycache__' not in f.parts and f.suffix!='.pyc' and str(f.relative_to(ROOT)) not in fingerprints.get('path_mapping',{})}
        assert set(manifest)==all_paths,(sorted(all_paths-set(manifest)),sorted(set(manifest)-all_paths))
        for name,metadata in manifest.items():
            path=ROOT/name
            assert path.stat().st_size==metadata['bytes'] and sha(path)==metadata['sha256'],name
        print('PASS: file inventory and all release SHA-256 checks')
    if args.check_tables:
        expected=json.loads((ROOT/'scripts/analysis_table_sha256.json').read_text())
        assert len(expected)>=9
        for name,h in expected.items():assert sha(ROOT/name)==h,name
        print(f'PASS: {len(expected)} stored analysis CSVs match byte-for-byte')
if __name__=='__main__':main()

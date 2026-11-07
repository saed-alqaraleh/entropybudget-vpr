#!/usr/bin/env python3
"""Create untracked compatibility paths without altering any original source bytes."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    record=json.loads((ROOT/'source_fingerprints.json').read_text())
    mapping=record['path_mapping']
    for old,new in mapping.items():
        src=ROOT/new
        dest=ROOT/old
        assert src.is_file(),new
        if dest.exists():
            assert dest.is_file() and digest(dest)==digest(src),f'Existing path differs: {old}'
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
        assert digest(dest)==digest(src)
    print('PASS: compatibility paths copied byte-for-byte; tracked sources unchanged')
if __name__=='__main__':main()

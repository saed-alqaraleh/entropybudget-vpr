"""Pure CPU tests: persistent manifests and pre-timing local bytes, not GPU benchmarks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vprbudget.data import load_dataset, _paths_from_manifest, _save_manifest_once
import benchmark_e2e_resilient as b

with TemporaryDirectory() as temp:
    p=Path(temp); db=p/'data'/'nordland'/'images'/'test'/'database';q=p/'data'/'nordland'/'images'/'test'/'queries'
    db.mkdir(parents=True);q.mkdir(parents=True)
    manifest=p/'data'/'nordland'/'all_images_paths.txt'
    with manifest.open('w') as f:
        for grp in ('database','queries'):
            for i in range(27592):
                f.write(f'images/test/{grp}/@0@{i:05d}.0@@@@@{i}@@@@@@@@.jpg\n')
    # Simulate exactly the DriveFS defect; the loader must never list the giant folder.
    with patch.object(Path,'iterdir',side_effect=OSError(5,'simulated DriveFS readdir EIO')):
        ds=load_dataset('nordland',p/'data')
    assert ds.n_database==ds.n_queries==27592
    assert ds.query_paths[34].name=='@0@00034.0@@@@@34@@@@@@@@.jpg'
    assert ds.database_paths[27591].name=='@0@27591.0@@@@@27591@@@@@@@@.jpg'
    assert ds.manifest_source==str(manifest)
    # Original manifest is idempotent, user data is never modified by loader.
    assert manifest.read_text().count('\n')==55184
    # Persist new dataset manifests in a dedicated Drive-like folder without changing dataset.
    d2=p/'new'/'baidu'/'images'/'test'/'database';q2=d2.parent/'queries';d2.mkdir(parents=True);q2.mkdir(parents=True)
    newroot=d2.parents[2]
    dbpaths=[d2/f'@0@{i:05d}.0@@@@@{i}@@@@@@@@.jpg' for i in range(5207)]
    qpaths=[q2/f'@0@{i:05d}.0@@@@@{i}@@@@@@@@.jpg' for i in range(3208)]
    saved=p/'drive_backup'/'baidu_all_images_paths.txt'
    _save_manifest_once(newroot,d2.parent,dbpaths,qpaths,saved)
    x,y=_paths_from_manifest(newroot,d2.parent,saved)
    assert len(x)==5207 and len(y)==3208
    _save_manifest_once(newroot,d2.parent,dbpaths,qpaths,saved)
    # Selected source images staged local BEFORE model / timing.
    for i in (0,3,5):
        fp=q/f'@0@{i:05d}.0@@@@@{i}@@@@@@@@.jpg';fp.write_bytes(b'raw-image'+bytes([i]))
    ds2=types.SimpleNamespace(name='nordland',query_paths=ds.query_paths.copy(),
                              database_paths=ds.database_paths[:100])
    rr={'query_indices':np.arange(27592)}
    report=b._stage_local_queries(ds2,rr,np.array([0,3,5]),p/'scratch')
    assert report['n_selected_queries']==3
    for i in (0,3,5):
        assert ds2.query_paths[i].read_bytes()==b'raw-image'+bytes([i])
        assert str(ds2.query_paths[i]).startswith(str(p/'scratch'))
    # Mirroring paid gallery tensor files should not recompute SuperPoint features.
    cfg=types.SimpleNamespace(detector='superpoint',matching_image_size=512,max_keypoints=2048)
    ca=np.tile(np.arange(100),(3,1))
    origin=p/'master';target=p/'fast_local'
    for i in range(50):
        pp=b._feature_cache_path(origin,ds2.database_paths[i],cfg)
        pp.parent.mkdir(parents=True,exist_ok=True);pp.write_bytes(f'cached-feature-{i}'.encode())
    stats=b._mirror_local_gallery_cache(ds2,ca,cfg,origin,target)
    assert stats['gallery_features_mirrored']==50
    assert stats['new_copies']==50
    stats2=b._mirror_local_gallery_cache(ds2,ca,cfg,origin,target)
    assert stats2['new_copies']==0
    # Unexpected/unsafe relative paths are rejected.
    unsafe=p/'unsafe.txt';unsafe.write_text('images/test/database/../../escape.jpg\n')
    try:_paths_from_manifest(p/'data'/'nordland',p/'data'/'nordland'/'images/test',unsafe)
    except Exception:pass
    else:raise AssertionError('Path traversal manifest was accepted')
print('PASS: giant-folder DriveFS EIO bypass; exact 27,592/27,592 manifest order; '
      'Baidu manifest replay; local raw-query staging; local feature-cache mirroring; unsafe-path guard')

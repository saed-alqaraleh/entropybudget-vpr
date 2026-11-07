"""Durable, checked, immutable-identity Drive checkpoint chunks for the E2E runner.

Operations run *after* per-query stage timing. A checkpointed run cannot claim
uninterrupted batch wall-time. The recorder does not run model inference.
"""
from __future__ import annotations
import hashlib, io, json, os, shutil, zipfile
from pathlib import Path


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda:src.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def put_checked(path,contents):
    """Write to mounted Drive, verify complete byte-level readback, fail closed."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.read_bytes()==contents:
        return 'already_identical'
    if path.exists():
        raise RuntimeError('Refusing to replace nonidentical durable evidence: '+str(path))
    tmp=path.with_name(path.name+'.incomplete')
    if tmp.exists():tmp.unlink() # never count incomplete file as a checkpoint
    with tmp.open('wb') as f:
        f.write(contents);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if path.read_bytes()!=contents:
        raise RuntimeError('Drive checkpoint readback differs from original: '+str(path))
    return 'saved_verified'


def copy_checked(src, dest):
    src=Path(src);dest=Path(dest)
    return put_checked(dest,src.read_bytes())


def checkpoint_block(method_dir,backup_method_dir,start,stop,spec_sha):
    """Zip [start,stop) JSON records into one immutable Drive file."""
    method_dir=Path(method_dir); backup_method_dir=Path(backup_method_dir)
    rows=[]; hashes={}
    for i in range(start,stop):
        name=f'query_{i:05d}.json'
        b=(method_dir/name).read_bytes()
        row=json.loads(b)
        if int(row['query_slot'])!=i:raise ValueError('Checkpoint slot mismatch: '+name)
        rows.append((name,b));hashes[name]=sha256_bytes(b)
    meta={'schema':'entropybudget-query-block-v2','spec_sha256':spec_sha,
          'start':start,'stop':stop,'sha256':hashes}
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        # Explicit fixed ZIP headers make rerun byte-for-byte deterministic.
        for name,b in [('BLOCK_MANIFEST.json',json.dumps(meta,sort_keys=True).encode())]+rows:
            zi=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            zi.compress_type=zipfile.ZIP_DEFLATED
            zi.external_attr=0o600 << 16
            z.writestr(zi,b)
    file=backup_method_dir/'chunks'/f'query_{start:05d}_{stop-1:05d}.zip'
    act=put_checked(file,out.getvalue())
    return file,act


def recover_blocks(backup_method_dir,method_dir,expected_spec_sha,expected_n):
    """Fail on tampering/overlap; restore only explicitly verified saved blocks."""
    backup_method_dir=Path(backup_method_dir)
    method_dir=Path(method_dir)
    seen={}
    for block in sorted((backup_method_dir/'chunks').glob('query_*.zip')):
        with zipfile.ZipFile(block) as z:
            bad=z.testzip()
            if bad:raise RuntimeError(f'Corrupt Drive chunk {block}: {bad}')
            meta=json.loads(z.read('BLOCK_MANIFEST.json'))
            if meta.get('schema')!='entropybudget-query-block-v2' or meta.get('spec_sha256')!=expected_spec_sha:
                raise RuntimeError(f'Drive checkpoint spec mismatch: {block}')
            a,b=int(meta['start']),int(meta['stop'])
            if not (0<=a<b<=expected_n):raise RuntimeError('Invalid chunk indices: '+str(block))
            expected=[f'query_{i:05d}.json' for i in range(a,b)]
            if sorted(meta['sha256'])!=sorted(expected):raise RuntimeError('Incorrect checkpoint manifest')
            if sorted(x for x in z.namelist() if x!='BLOCK_MANIFEST.json')!=sorted(expected):
                raise RuntimeError('Unexpected zip contents: '+str(block))
            for i,name in enumerate(expected,a):
                raw=z.read(name)
                if sha256_bytes(raw)!=meta['sha256'][name]:raise RuntimeError('Checkpoint file hash mismatch')
                row=json.loads(raw)
                if row.get('query_slot')!=i or row.get('run_spec_sha256')!=expected_spec_sha:
                    raise RuntimeError('Checkpoint row metadata mismatch')
                if i in seen and seen[i]!=raw:raise RuntimeError('Conflicting overlapping checkpoint rows')
                seen[i]=raw
    method_dir.mkdir(parents=True,exist_ok=True)
    for i,raw in seen.items():
        path=method_dir/f'query_{i:05d}.json'
        if path.is_file() and path.read_bytes()!=raw:raise RuntimeError('Local and Drive checkpoint differ: '+str(path))
        if not path.exists():path.write_bytes(raw)
    return len(seen)


def persist_artifact(local_root,backup_root,relative_path):
    path=Path(relative_path)
    return copy_checked(Path(local_root)/path,Path(backup_root)/path)


def stable_spec(args,dsname,rep,method,environment,selected):
    """Identity fingerprint includes frozen protocol and model/source versions."""
    info={
        'schema':'entropybudget-e2e-v2',
        'dataset':str(dsname),'method':str(method),'repetition':int(rep),
        'selected_positions':[int(x) for x in selected],
        'mode':args.mode,'n_queries':int(args.n_queries),
        'seed':int(args.seed),'feature_mode':args.feature_mode,
        'topk':100,'entropy_T':.10,'selected_fraction':.4,'selected_depth':50,
        'device':environment['device'],'torch':environment['torch'],'cuda':environment['cuda'],
        'gpu_driver':environment['gpu_driver'],'lightglue_commit':environment['lightglue_commit'],
        'checkpoint_hashes':environment['checkpoint_hashes'],
        'source_hashes':environment['source_hashes'],
        'raw_query_image_io':environment.get('raw_query_image_io'),
        'gallery_feature_io':environment.get('gallery_feature_io'),
        'query_image_set_sha256':environment.get('query_image_set_sha256'),
        'boq_repo_fingerprint':environment.get('boq_source_recovery',{}).get('hubconf_sha256'),
        'boq_repo_commit':environment.get('boq_source_recovery',{}).get('repository_commit'),
    }
    blob=json.dumps(info,sort_keys=True,default=str).encode()
    return sha256_bytes(blob), info


def persist_first_provenance(local_root,backup_root,relative_path):
    """Keep the first GPU preparation/warmup record; later sessions are different."""
    target=Path(backup_root)/Path(relative_path)
    if target.exists():
        if not target.is_file():raise RuntimeError('Backup artifact is not a file: '+str(target))
        return 'first_session_original_retained'
    return persist_artifact(local_root,backup_root,relative_path)

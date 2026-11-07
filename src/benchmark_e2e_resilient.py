#!/usr/bin/env python3
"""EntropyBudget true *batch-workload* image-to-ranking latency experiment.

A clean implementation adapter using recovered historical functions. GPU benchmark
must be run by the author in Colab on the actual images and historical gallery.

Measurement modes:
  subset_calibrated: all N measured queries undergo fresh descriptor inference
    + retrieval. Entropy percentile is built from these N measured queries,
    so this is a NEW N-query workload (not the historical full-dataset policy).
  original_full_cohort_replay: reproduce original N_full-derived budgets for the
    historic sample; that calibration is ARCHIVED and excluded from timing.
    This mode is conditional service timing, not full calibrated E2E.

Gallery descriptor index and (optionally) gallery local features are prepared
outside the measurement; query descriptors and query local features are never
read from cache within a timed trial. Decoding and preprocessing ARE timed.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, platform, random, shutil, subprocess, sys, time, uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import persistent_checkpoint as durable

FROZEN_TOPK=100
FROZEN_T=0.10
FROZEN_HIGH_FRACTION=.40
FROZEN_DEPTH=50
DATASET_ORDER=('st_lucia','nordland','baidu')
POLICIES=('uniform_m20','entropy_budget','uniform_m50')

def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()

def _atomic_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,default=str)+'\n',encoding='utf8')
    os.replace(temp,p)

def _load_py(p,name):
    spec=importlib.util.spec_from_file_location(name,str(p))
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

def _sync(torch):
    if torch.cuda.is_available():torch.cuda.synchronize()

def _now():return datetime.now(timezone.utc).isoformat()

def _shard_dirs(workspace,ds):
    p=workspace/'q1_external_confirmatory_v3'/'boq-dinov2'/'descriptors'/ds
    fs=sorted(p.glob('database_*.npz'))
    if not fs:raise FileNotFoundError('Missing original BoQ-DINOv2 gallery descriptor shards: '+str(p))
    return fs

def _make_faiss_index(shards, n_gallery):
    import faiss
    index=None
    counts=[]
    for f in shards:
        with np.load(f) as z:
            vec=np.ascontiguousarray(z['descriptors'],dtype=np.float32)
        if vec.ndim != 2 or not np.isfinite(vec).all():raise RuntimeError('Bad gallery shard '+str(f))
        if index is None:index=faiss.IndexFlatL2(vec.shape[1])
        if vec.shape[1]!=index.d:raise RuntimeError('Gallery descriptor dimensionality changed')
        index.add(vec);counts.append(len(vec));del vec
    if index.ntotal != n_gallery:raise RuntimeError(f'Gallery/index size mismatch {index.ntotal} != {n_gallery}')
    return index,counts

def _cache_id(path,config):
    # Source-image path alone was used by historical code; include frozen image
    # processing settings as an additional cache guard, not a scientific change.
    key=f'{str(Path(path).resolve())}|{config.detector}|{config.matching_image_size}|{config.max_keypoints}'
    return hashlib.sha256(key.encode()).hexdigest()

def _cpu_tree(v):
    import torch
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:_cpu_tree(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return type(v)(_cpu_tree(x) for x in v)
    return v

def _device_tree(v,device):
    import torch
    if torch.is_tensor(v):return v.to(device,non_blocking=True)
    if isinstance(v,dict):return {k:_device_tree(x,device) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return type(v)(_device_tree(x,device) for x in v)
    return v

def _feature_cache_path(cache_dir,image_path,cfg):
    h=_cache_id(image_path,cfg)
    return cache_dir/h[:2]/f'{h}.pt'

def _prepare_features(extractor,ds,retrieval,positions,cfg,cache_dir,device,matcher,limit=None,live_candidate_positions=None):
    """Checkpointed offline gallery local features only. Query features excluded."""
    import torch
    from vprbudget.matching import extract_features
    if live_candidate_positions is None:
        raise RuntimeError('Refusing historical-only gallery prebuild; first compute LIVE retrieval candidates')
    cand=np.asarray(live_candidate_positions,dtype=np.int64)
    if cand.shape!=(len(positions),FROZEN_TOPK):
        raise ValueError(f'Invalid LIVE candidate matrix {cand.shape}; expected ({len(positions)},100)')
    ids=set(int(x) for x in cand[:,:FROZEN_DEPTH].ravel())
    ids=sorted(ids)
    if limit is not None:ids=ids[:limit]
    new=0
    for i,idx in enumerate(ids):
        im=ds.database_paths[idx]
        p=_feature_cache_path(cache_dir,im,cfg)
        if p.is_file():continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():feats,wh=extract_features(extractor,im,cfg.matching_image_size,device)
        obj={'feats':_cpu_tree(feats),'wh':tuple(map(int,wh)),'config':{'detector':cfg.detector,'kp':cfg.max_keypoints,'size':cfg.matching_image_size}}
        temp=p.with_name(p.name+'.tmp')
        torch.save(obj,temp);os.replace(temp,p)
        new+=1
        if new%100==0:print(f'  gallery local features: created {new}, scanned {i+1}/{len(ids)}',flush=True)
    return dict(union_images=len(ids),new_images=new,already_cached=len(ids)-new)

def _live_candidate_preflight(ctx,ds,ret,positions,index,out,backup_root):
    """Untimed fresh-BoQ retrieval for exact gallery cache preparation.

    This is NOT the timed query prepass; the measured call recomputes each
    query descriptor and retrieval independently. Do not relabel as E2E time.
    """
    model=ctx['bo_q'];torch=ctx['torch'];device=ctx['device']
    transform=ctx['hist'].official_input_transform()
    qidx=ret['query_indices'].astype(int)
    candidates=np.empty((len(positions),FROZEN_TOPK),dtype=np.int64)
    for j,qp in enumerate(positions):
        q=int(qidx[int(qp)]);img=ds.query_paths[q]
        desc=_model_descriptor(model,img,transform,device,torch)
        _,indices=index.search(np.ascontiguousarray(desc[None,:],dtype=np.float32),FROZEN_TOPK)
        candidates[j]=indices[0]
    target=out/'offline_live_preflight'/f'{ds.name}_candidate_ids.npy'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        old=np.load(target)
        if not np.array_equal(old,candidates):
            raise RuntimeError('LIVE preflight changes candidate identities versus the prior run. '
                               'Keep original evidence and start a NEW run identifier.')
    else:
        tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as f:np.save(f,candidates)
        os.replace(tmp,target)
    # This matrix is audit metadata. It contains no cached query feature tensors.
    durable.persist_artifact(out,backup_root,target.relative_to(out))
    exact=float(np.mean(np.all(candidates==ret['candidates'][positions],axis=1)))
    report={'dataset':ds.name,'cohort':len(positions),
            'live_top50_gallery_union':len(np.unique(candidates[:,:FROZEN_DEPTH])),
            'archived_top100_agreement':exact,
            'candidate_matrix_sha256':_sha(target),
            'note':'UNTIMED fresh-image preflight for gallery-local cache coverage, not a timed inference result'}
    _atomic_json(out/'offline_live_preflight'/f'{ds.name}.json',report)
    durable.persist_artifact(out,backup_root,Path('offline_live_preflight')/f'{ds.name}.json')
    return candidates,report


def _assert_cache_coverage(ds,cache_dir,cfg,candidates,tag):
    """Fail before any local timing if a LIVE paid candidate is missing."""
    if np.asarray(candidates).shape[1] != FROZEN_TOPK:
        raise ValueError('Live candidate matrix must have top-100 entries')
    ids=np.unique(np.asarray(candidates)[:,:FROZEN_DEPTH])
    missing=[]
    for idx in ids:
        p=_feature_cache_path(cache_dir,ds.database_paths[int(idx)],cfg)
        if not p.is_file():missing.append(str(p))
    if missing:
        raise RuntimeError(f'{ds.name} {tag}: {len(missing)} LIVE top-50 gallery feature files missing; '
                           f'first={missing[0]}. Do not begin any measured policy.')
    return int(len(ids))


def _warm_query_storage(ds,ret,positions):
    """Equalize file-system coldness outside timing, once before each policy."""
    from PIL import Image
    for qp in positions:
        qidx=int(ret['query_indices'][int(qp)])
        with Image.open(ds.query_paths[qidx]) as im:
            im.convert('RGB').load()


def _load_feature(path,cfg,cache_dir,device):
    import torch
    p=_feature_cache_path(cache_dir,path,cfg)
    if not p.is_file():raise FileNotFoundError('Offline gallery-local cache incomplete (re-ranked retrieval may differ): '+str(p))
    obj=torch.load(p,map_location='cpu',weights_only=True)
    if obj['config']!={'detector':cfg.detector,'kp':cfg.max_keypoints,'size':cfg.matching_image_size}:
        raise RuntimeError('Gallery local feature cache configuration mismatch: '+str(p))
    return _device_tree(obj['feats'],device),tuple(obj['wh'])

def _copy_local_retry(src, dest, attempts=4):
    """Best-effort bounded DriveFS copies before any timed measurement."""
    import io
    src=Path(src);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and dest.stat().st_size>0:return 'reused'
    last=None
    for attempt in range(attempts):
        temp=dest.with_name(dest.name+'.incomplete')
        try:
            if temp.exists():temp.unlink()
            with src.open('rb') as f, temp.open('wb') as w:
                shutil.copyfileobj(f,w,length=1024*1024)
                w.flush();os.fsync(w.fileno())
            if temp.stat().st_size!=src.stat().st_size or temp.stat().st_size==0:
                raise IOError('Image/feature file short-copy: '+str(src))
            os.replace(temp,dest)
            return 'copied'
        except (OSError, IOError) as exc:
            last=exc
            if attempt<attempts-1:time.sleep(1.5*(attempt+1))
    raise RuntimeError(f'DriveFS copy failed before GPU measurement; original={src}; '
                       f'last_error={last}. The Google Drive mount may need repair.') from last


def _stage_local_queries(ds,ret,positions,out):
    """Transfer selected raw images to Colab scratch, NOT cached query descriptors.

    Keeps query images local during online timing, removing repeated small-file
    reads from a giant DriveFS image directory. Preserves the frozen query IDs.
    """
    qidx=ret['query_indices'].astype(int)
    unique=sorted(set(int(qidx[int(i)]) for i in positions))
    staged=0
    image_provenance=[]
    for i in unique:
        original=ds.query_paths[i]
        dest=out/'offline_raw_image_stage'/ds.name/'queries'/f'{i:07d}_{original.name}'
        act=_copy_local_retry(original,dest)
        if act=='copied':staged+=1
        ds.query_paths[i]=dest
        image_provenance.append({'query_index':i,'original_name':original.name,
                                 'bytes':dest.stat().st_size,'sha256':_sha(dest)})
    content_sha=hashlib.sha256(json.dumps(image_provenance,sort_keys=True).encode()).hexdigest()
    print(f'RAW IMAGE LOCAL STAGING: {ds.name}: {len(unique)} selected images; '
          f'new={staged}, reused={len(unique)-staged}',flush=True)
    return {'n_selected_queries':len(unique),'new_copies':staged,
            'content_manifest_sha256':content_sha,'images':image_provenance,
            'local_raw_images':True,'timing_boundary':'OFFLINE before GPU policy timing; raw pixels unchanged'}


def _mirror_local_gallery_cache(ds,candidates,cfg,origin,destination):
    """Mirror paid gallery feature tensors once so the timed matching loop avoids DriveFS."""
    ids=np.unique(np.asarray(candidates)[:,:FROZEN_DEPTH])
    copied=0
    for j,idx in enumerate(ids):
        original=ds.database_paths[int(idx)]
        src=_feature_cache_path(origin,original,cfg)
        dst=_feature_cache_path(destination,original,cfg)
        if not src.is_file():raise FileNotFoundError('Offline master cache missing: '+str(src))
        if _copy_local_retry(src,dst)=='copied':copied+=1
        if (j+1)%500==0:print(f'  local gallery-cache mirror {j+1}/{len(ids)}',flush=True)
    _assert_cache_coverage(ds,destination,cfg,candidates,'LOCAL GALLERY MIRROR')
    return {'gallery_features_mirrored':int(len(ids)),'new_copies':copied,
            'note':'Source cache remains in persistent Drive; local cache used for timed matching only'}


def _cpu_preflight(ws,datasets,n_queries,seed,manifest_dir):
    """Metadata-first preflight. A DriveFS error is reported at its exact stage.

    Only one selected raw query image is copied per dataset to local scratch
    and decoded there. No GPU work and no large-directory listing is needed
    when native/verified portable filename manifests are present.
    """
    from vprbudget.data import load_dataset
    from PIL import Image
    print('PREFLIGHT STAGE: frozen subset archive',flush=True)
    selected=_frozen_subsets(ws,datasets,n_queries,seed)
    for name in datasets:
        print(f'PREFLIGHT STAGE: {name}: dataset manifest and coordinates',flush=True)
        try:
            ds=load_dataset(name,ws/'data_external_q1_v3',manifest_dir=manifest_dir)
        except Exception as exc:
            raise RuntimeError(f'{name}: dataset manifest/coordinate load failed; '
                               f'check Colab Drive mount. Underlying exception: {type(exc).__name__}: {exc}') from exc
        print(f'PREFLIGHT STAGE: {name}: archived retrieval',flush=True)
        rp=ws/'q1_external_confirmatory_v3'/'boq-dinov2'/'retrieval'/f'{name}.npz'
        with np.load(rp) as z:
            qidx=z['query_indices'];cand=z['candidates']
        if len(qidx)!=ds.n_queries or cand.shape[1]!=100 or np.max(cand)>=ds.n_database:
            raise RuntimeError(f'{name}: dataset/gallery identity differs from archived retrieval')
        print(f'PREFLIGHT STAGE: {name}: sample raw image read',flush=True)
        loc=int(qidx[int(selected[name][0])]);image=ds.query_paths[loc]
        if not image.name.lower().endswith(('.jpg','.jpeg','.png')):
            raise RuntimeError(f'Unexpected image name {image.name}')
        # DriveFS can transiently fail even for a single read. Copy once with
        # bounded retries, and decode from local scratch rather than Drive.
        sample=Path('/content/entropybudget_cpu_preflight_sample')/name/image.name
        try:
            _copy_local_retry(image,sample,attempts=3)
            with Image.open(sample) as im:im.verify()
        except Exception as exc:
            raise RuntimeError(f'{name}: selected image DriveFS read/decode failed: {image}; '
                               f'original={type(exc).__name__}: {exc}. If errno 5, '
                               'remount Google Drive; do not rerun GPU models.') from exc
        print(f'PREFLIGHT STAGE: {name}: gallery descriptor shard metadata',flush=True)
        shard_count=len(_shard_dirs(ws,name))
        print('DATA PREFLIGHT PASS',name,'gallery',ds.n_database,'queries',ds.n_queries,
              'frozen_sample',len(selected[name]),'manifest',getattr(ds,'manifest_source',''),
              'descriptor_shards',shard_count,'first_query',image.name,flush=True)
    print('ALL DATA PREFLIGHTS PASS: no GPU models loaded',flush=True)


def _load_models_and_sources(pkg,workspace):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for measured E2E benchmark. Use Colab GPU runtime.')
    sys.path.insert(0,str(pkg/'src'))
    from vprbudget.config import load_config
    from vprbudget.matching import extract_features,match_pair,geometric_verify
    from vprbudget.scoring import global_score,global_reliability,spatial_coverage,spatial_distribution_score
    from lightglue import SuperPoint,LightGlue
    hist=_load_py(pkg/'src/Q1_External_Confirmatory_Runner_v3.py','historical_runner')
    safe=_load_py(pkg/'src/Q1_Budget_Safe_Allocator_v1.py','historical_safe')
    cfg=load_config(str(pkg/'internal_provenance/config_q1_external_superpoint_v3.json'))
    if (str(cfg.detector).lower(),str(cfg.matcher).lower()) != ('superpoint','lightglue'):
        raise ValueError('Frozen primary local backend must be SuperPoint/LightGlue')
    device=torch.device('cuda')
    # Recover unchanged official BoQ hubconf locally; bypass GitHub default-branch
    # discovery that can raise HTTP 504 even before any inference.
    from boq_hub_recovery import load_historical_boq_wrapper
    bo_q,source,boq_provenance=load_historical_boq_wrapper(hist,device,workspace)
    extractor=SuperPoint(max_num_keypoints=cfg.max_keypoints).eval().to(device)
    lg=LightGlue(features='superpoint').eval().to(device)
    return locals()

def _env_info(workspace,config,model,boq_provenance=None):
    import torch,cv2,faiss
    cmds={}
    for cmd,key in [(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],'gpu_driver'),
                    (['git','-C','/content/LightGlue','rev-parse','HEAD'],'lightglue_commit')]:
        try:cmds[key]=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
        except Exception as exc:cmds[key]='UNVERIFIED: '+str(exc)
    ckpt=Path.home()/'.cache/torch/hub/checkpoints'
    models={str(p.name):_sha(p) for p in ckpt.glob('*.pth') if p.is_file()}
    try: git_hub=subprocess.check_output(['git','-C',str(Path.home()/'.cache/torch/hub/amaralibey_bag-of-queries_main'),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:git_hub='UNVERIFIED (torch hub cache branch may differ)'
    return dict(timestamp_utc=_now(),device=str(torch.cuda.get_device_name(0)),torch=torch.__version__,cuda=torch.version.cuda,
                cv2=cv2.__version__,faiss=getattr(faiss,'__version__','unknown'),numpy=np.__version__,python=sys.version,
                os=platform.platform(),cpu_threads=torch.get_num_threads(),checkpoint_hashes=models,boq_repo_commit=git_hub,
                boq_source_recovery=boq_provenance,**cmds)

def _model_descriptor(model,path,transform,device,torch):
    from PIL import Image
    # All operations here fall inside the independently measured global path.
    with Image.open(path) as im: img=transform(im.convert('RGB'))
    x=img[None].to(device)
    with torch.inference_mode(): v=model(x)
    _sync(torch)
    out=v.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    if not np.isfinite(out).all():raise RuntimeError('Global descriptor NaN/inf at '+str(path))
    return out

def _query_local_pipeline(image_path,candidates,ds,budget,ctx,G,cg,cache_dir,feature_mode):
    """All paid image/local operations run for exactly m=budget candidates."""
    torch=ctx['torch'];cfg=ctx['cfg'];device=ctx['device']
    from vprbudget.matching import extract_features, match_pair, geometric_verify
    from vprbudget.scoring import spatial_coverage,spatial_distribution_score
    safe=ctx['safe']
    n=np.zeros((1,FROZEN_TOPK),np.int32);mc=np.zeros_like(n);sp=np.zeros((1,FROZEN_TOPK),np.float32)
    stages={'query_local_decode_extract_ms':0.,'gallery_local_access_ms':0.,'lightglue_matching_ms':0.,
            'magsac_spatial_ms':0.,'fixed_scoring_ms':0.}
    if budget>0:
        _sync(torch); t=time.perf_counter()
        with torch.inference_mode():qfeat,wq=extract_features(ctx['extractor'],image_path,cfg.matching_image_size,device)
        _sync(torch);stages['query_local_decode_extract_ms']=(time.perf_counter()-t)*1000
        for k in range(int(budget)):
            idx=int(candidates[k]);gallery_image=ds.database_paths[idx]
            _sync(torch);t=time.perf_counter()
            if feature_mode=='prebuilt':
                cfeat,wc=_load_feature(gallery_image,cfg,cache_dir,device)
            else:
                with torch.inference_mode():cfeat,wc=extract_features(ctx['extractor'],gallery_image,cfg.matching_image_size,device)
            _sync(torch);stages['gallery_local_access_ms']+=(time.perf_counter()-t)*1000
            _sync(torch);t=time.perf_counter()
            kpts0,kpts1=match_pair(ctx['extractor'],ctx['lg'],qfeat,cfeat,device)
            _sync(torch);stages['lightglue_matching_ms']+=(time.perf_counter()-t)*1000
            mc[0,k]=len(kpts0)
            t=time.perf_counter()
            mask=geometric_verify(kpts0,kpts1,cfg.geometric_model,cfg.ransac_reproj_threshold,
                                  cfg.ransac_confidence,cfg.ransac_max_iters,cfg.min_matches_for_gv)
            ni=int(mask.sum());n[0,k]=ni
            if ni:
                bq,cq=spatial_coverage(kpts0[mask],wq,cfg.spatial_grid)
                bc,cc=spatial_coverage(kpts1[mask],wc,cfg.spatial_grid)
                sp[0,k]=spatial_distribution_score(bq,bc,cq,cc)
            stages['magsac_spatial_ms']+=(time.perf_counter()-t)*1000
            del cfeat,kpts0,kpts1
        del qfeat
    t=time.perf_counter()
    top=int(safe.budget_safe_top1(G[None,:],n,mc,sp,np.asarray([cg]),budget,cfg)[0])
    stages['fixed_scoring_ms']=(time.perf_counter()-t)*1000
    # Real matching path paid head checked: no matching beyond budget.
    assert len(candidates)==FROZEN_TOPK and not np.any(mc[0,int(budget):])
    for mode in ('zeros','adversarial'):
        if int(budget)==100:continue
        n2=n.copy();mc2=mc.copy();sp2=sp.copy()
        if mode=='adversarial':
            n2[:,int(budget):]=1000000;mc2[:,int(budget):]=1000000;sp2[:,int(budget):]=1
        else:
            n2[:,int(budget):]=0;mc2[:,int(budget):]=0;sp2[:,int(budget):]=0
        if int(safe.budget_safe_top1(G[None,:],n2,mc2,sp2,np.asarray([cg]),budget,cfg)[0])!=top:
            raise RuntimeError('Production local-head noninterference failed')
    return top,stages

def _frozen_subsets(workspace,datasets,nmax,seed):
    # Historical Phase-D draws one RNG across datasets in fixed order.
    rng=np.random.default_rng(seed); out={}
    for ds in DATASET_ORDER:
        retpath=workspace/'q1_external_confirmatory_v3'/'boq-dinov2'/'retrieval'/f'{ds}.npz'
        if not retpath.is_file():raise FileNotFoundError('Missing frozen external retrieval: '+str(retpath))
        with np.load(retpath) as ret:n=len(ret['query_indices'])
        out[ds]=np.sort(rng.choice(n,size=min(n,nmax),replace=False)).astype(int)
    return {ds:out[ds] for ds in datasets}


def _hardware_warmup(ctx, ds, ret, positions, index, cfg, cache_dir, feature_mode, warmup_count):
    """Run genuine model/geometry/scorer kernels outside recorded timing."""
    from vprbudget.scoring import global_score,global_reliability
    torch=ctx['torch'];device=ctx['device']
    transform=ctx['hist'].official_input_transform()
    n=min(int(warmup_count),len(positions))
    start=time.perf_counter()
    for qp in positions[:n]:
        qidx=int(ret['query_indices'][int(qp)])
        image=ds.query_paths[qidx]
        desc=_model_descriptor(ctx['bo_q'],image,transform,device,torch)
        D,I=index.search(np.ascontiguousarray(desc[None,:],dtype=np.float32),FROZEN_TOPK)
        d=np.sqrt(np.maximum(D[0],0)).astype(float)
        cand=I[0].astype(np.int64)
        G=global_score(d,cfg.beta_global)
        cg=global_reliability(d,cfg.softmax_temperature,cfg.global_reliability_weights)
        # Use paid top-1 for warm-up: the branch calls real SuperPoint, LightGlue,
        # MAGSAC, and scorer. This is an outside-timer kernel warm-up, not a claim
        # of running all 50 matching operations per warm-up query.
        _query_local_pipeline(image,cand,ds,1,ctx,G,cg,cache_dir,feature_mode)
    _sync(torch)
    return {'warmup_queries':int(n),'matching_pairs':int(n),
            'warmup_wall_ms':float((time.perf_counter()-start)*1000),
            'note':'one paid candidate per warm-up query; not 50-candidate policy warm-up'}

def _run_one(ctx, ds, original_ret, positions, index, cfg, cache_dir, out_root, method, rep, args, report):
    import torch
    from vprbudget.scoring import global_score,global_reliability
    import pandas as pd
    runner=ctx['hist'];model=ctx['bo_q'];transform=runner.official_input_transform();device=ctx['device']
    archive_query_idx=original_ret['query_indices'].astype(int)
    method_dir=out_root/'records'/ds.name/f'rep_{rep:02d}'/method
    backup_method_dir=Path(args.drive_backup_root)/'records'/ds.name/f'rep_{rep:02d}'/method
    spec_sha,identity=durable.stable_spec(args,ds.name,rep,method,report,positions)
    spec_path=out_root/'specs'/ds.name/f'rep_{rep:02d}'/f'{method}.json'
    _atomic_json(spec_path,{'spec_sha256':spec_sha,'identity':identity})
    durable.persist_artifact(out_root,Path(args.drive_backup_root),spec_path.relative_to(out_root))
    restored=durable.recover_blocks(backup_method_dir,method_dir,spec_sha,len(positions))
    if restored:print(f'  RESTORED {restored} verified Drive query records: {ds.name} {method}',flush=True)
    if method_dir.exists() and any(method_dir.glob('query_*.json')) and not (args.overwrite_run or args.resume_partial):
        raise RuntimeError('Nonempty recorded measurement directory. Use verified --resume-partial: '+str(method_dir))
    if args.overwrite_run:raise RuntimeError('Unsafe overwrite unsupported in persistent checkpoint edition')
    method_dir.mkdir(parents=True,exist_ok=True)
    N=len(positions);dist=np.empty((N,100),dtype=np.float64);cand=np.empty((N,100),dtype=np.int64)
    global_ms=np.zeros(N);search_ms=np.zeros(N)
    archive_top1_match=np.zeros(N,dtype=bool);archive_top100_match=np.zeros(N,dtype=bool)
    t_batch=time.perf_counter()
    for j,qp in enumerate(positions):
        qid=int(archive_query_idx[qp]);image=ds.query_paths[qid]
        _sync(torch);t=time.perf_counter()
        desc=_model_descriptor(model,image,transform,device,torch)
        global_ms[j]=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        D,I=index.search(np.ascontiguousarray(desc[None,:],dtype=np.float32),FROZEN_TOPK)
        search_ms[j]=(time.perf_counter()-t)*1000
        dist[j]=np.sqrt(np.maximum(D[0],0)).astype(np.float64);cand[j]=I[0].astype(np.int64)
        archive_top1_match[j]=bool(cand[j,0]==int(original_ret['candidates'][qp,0]))
        archive_top100_match[j]=bool(np.array_equal(cand[j],original_ret['candidates'][qp]))
    prepass_wall_ms=(time.perf_counter()-t_batch)*1000
    # If live retrieval changes, fail before any local inference; never use a
    # historical top-50 union as the sole source of prebuilt local features.
    if args.feature_mode=='prebuilt':
        _assert_cache_coverage(ds,cache_dir,cfg,cand,'MEASURED PREPASS')
    expected_preflight=out_root/'offline_live_preflight'/f'{ds.name}_candidate_ids.npy'
    if expected_preflight.is_file():
        previous=np.load(expected_preflight)
        if not np.array_equal(cand,previous):
            raise RuntimeError(f'{ds.name}: live retrieval changed since untimed cache preflight; '
                               'no policy timing was started. Preserve artifacts, investigate source consistency.')
    # Entire batch prepass (fresh image-to-query descriptor + retrieval) precedes allocation.
    t_sched=time.perf_counter()
    if args.mode=='subset_calibrated':
        e=runner.retrieval_entropy(dist,FROZEN_T)
        high=runner.stable_top_fraction(e,FROZEN_HIGH_FRACTION,largest=True)
        frozen_budgets=np.where(high,FROZEN_DEPTH,0)
        calibrating_full_current_workload=True
    else:
        H=runner.retrieval_entropy(original_ret['descriptor_distances'].astype(float),FROZEN_T)
        frozen_budgets=np.where(runner.stable_top_fraction(H,FROZEN_HIGH_FRACTION,True)[positions],50,0)
        calibrating_full_current_workload=False
    scheduler_wall_ms=(time.perf_counter()-t_sched)*1000
    budgets={'uniform_m20':np.full(N,20,int),'entropy_budget':frozen_budgets,'uniform_m50':np.full(N,50,int)}[method]
    # All methods process the same physical query set and each computes its own fresh global prepass.
    rows=[]
    reused_existing=0
    for j,qp in enumerate(positions):
        qid=int(archive_query_idx[qp]);image=ds.query_paths[qid]
        G=global_score(dist[j],cfg.beta_global)
        cg=global_reliability(dist[j],cfg.softmax_temperature,cfg.global_reliability_weights)
        b=int(budgets[j])
        old_file=method_dir/f'query_{j:05d}.json'
        if args.resume_partial and old_file.is_file():
            row=json.loads(old_file.read_text())
            required={'dataset':ds.name,'method':method,'query_position':int(qp),'query_index':qid,'budget':b}
            if any(row.get(k)!=v for k,v in required.items()) or row.get('run_spec_sha256')!=spec_sha or row.get('query_slot')!=j:
                raise RuntimeError('Cannot resume: saved query identity/source/policy differs: '+str(old_file))
            new_hash=hashlib.sha256(np.asarray(cand[j],dtype=np.int64).tobytes()).hexdigest()
            if row.get('live_top100_identity_sha256')!=new_hash:
                raise RuntimeError('Cannot resume: live top100 candidate identity differs: '+str(old_file))
            rows.append(row);reused_existing+=1
            if (j+1)%args.checkpoint_every==0 or (j+1)==N:
                s=(j//args.checkpoint_every)*args.checkpoint_every
                p,act=durable.checkpoint_block(method_dir,backup_method_dir,s,j+1,spec_sha)
                print(f'  VERIFIED DRIVE CHECKPOINT {ds.name} {method}: {j+1}/{N} ({act})',flush=True)
            continue
        t_local=time.perf_counter()
        best,local_stages=_query_local_pipeline(image,cand[j],ds,b,ctx,G,cg,cache_dir,args.feature_mode)
        local_total_ms=(time.perf_counter()-t_local)*1000
        # Direct observed completion since start of method's batch, includes all waiting.
        completion_ms=(time.perf_counter()-t_batch)*1000
        row=dict(dataset=ds.name,method=method,repetition=rep,mode=args.mode,query_slot=j,
                 run_spec_sha256=spec_sha,query_position=int(qp),query_index=qid,
                 budget=b,verified_pairs=b,top1_candidate_position=best,top1_candidate_identity=int(cand[j,best]),
                 original_top1_same=bool(archive_top1_match[j]),original_top100_same=bool(archive_top100_match[j]),
                 live_top100_identity_sha256=hashlib.sha256(np.asarray(cand[j],dtype=np.int64).tobytes()).hexdigest(),
                 global_image_descriptor_ms=float(global_ms[j]),top100_search_ms=float(search_ms[j]),
                 local_path_ms=float(local_total_ms),
                 **{k:float(v) for k,v in local_stages.items()},
                 conditional_service_ms=float(global_ms[j]+search_ms[j]+local_total_ms),
                 direct_batch_completion_ms=float(completion_ms),
                 completion_includes_checkpoint_pauses=True,
                 timing_scope='fresh per-query descriptor+retrieval+local matching+GV; batch wall is interrupted for durable backup',
                 current_workload_entropy_calibration=calibrating_full_current_workload)
        # Never report a sum of p95; record directly measured per-query service/completion.
        rows.append(row)
        _atomic_json(method_dir/f'query_{j:05d}.json',row)
        if (j+1)%args.checkpoint_every==0 or (j+1)==N:
            s=(j//args.checkpoint_every)*args.checkpoint_every
            p,act=durable.checkpoint_block(method_dir,backup_method_dir,s,j+1,spec_sha)
            print(f'  VERIFIED DRIVE CHECKPOINT {ds.name} {method}: {j+1}/{N} ({act})',flush=True)
        if (j+1)%10==0:print(f'   {ds.name} {method}: {j+1}/{N}',flush=True)
    batch_wall_ms=(time.perf_counter()-t_batch)*1000
    summary={
      'dataset':ds.name,'method':method,'repetition':rep,'n':N,'mode':args.mode,
      'mean_budget':float(np.mean(budgets)),'total_verified_pairs':int(budgets.sum()),
      'prepass_wall_ms':float(prepass_wall_ms),'scheduler_wall_ms':float(scheduler_wall_ms),
      'complete_batch_wall_ms':None,'complete_batch_throughput_qps':None,
      'observed_wall_including_backup_pauses_ms':float(batch_wall_ms),
      'reused_previous_query_timings':reused_existing,'cross_fragment_e2e_valid':False,
      'batch_timing_status':'NOT_REPORTABLE: query checkpoints to Drive interrupt workload timeline',
      'archived_retrieval_top1_agreement':float(archive_top1_match.mean()),
      'archived_retrieval_exact_top100_agreement':float(archive_top100_match.mean()),
      'service_mean_ms':float(np.mean([r['conditional_service_ms'] for r in rows])),
      'service_median_ms':float(np.median([r['conditional_service_ms'] for r in rows])),
      'service_p95_ms':float(np.percentile([r['conditional_service_ms'] for r in rows],95)),
      'completion_mean_ms':float(np.mean([r['direct_batch_completion_ms'] for r in rows])) if not reused_existing else None,
      'completion_median_ms':float(np.median([r['direct_batch_completion_ms'] for r in rows])) if not reused_existing else None,
      'completion_p95_ms':float(np.percentile([r['direct_batch_completion_ms'] for r in rows],95)) if not reused_existing else None,
      'stage_mean_ms':{k:float(np.mean([r[k] for r in rows])) for k in ['global_image_descriptor_ms','top100_search_ms',
           'query_local_decode_extract_ms','gallery_local_access_ms','lightglue_matching_ms','magsac_spatial_ms','fixed_scoring_ms']},
      'note':'Completion times are measured from batch start; service times exclude workload waiting. Gallery descriptors/index (and optional gallery local features) are offline.'}
    _atomic_json(out_root/'summaries'/f'{ds.name}_rep{rep:02d}_{method}.json',summary)
    pd.DataFrame(rows).to_csv(method_dir/'query_timings.csv',index=False)
    durable.persist_artifact(out_root,Path(args.drive_backup_root),method_dir.relative_to(out_root)/'query_timings.csv')
    # The summary is last: presence means all method rows and CSV survived.
    durable.persist_artifact(out_root,Path(args.drive_backup_root),Path('summaries')/f'{ds.name}_rep{rep:02d}_{method}.json')
    return summary

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',required=True,help='Workspace containing the prepared benchmark inputs')
    ap.add_argument('--package-root',default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--output',required=True,help='New outputs directory; use /content for fastest timing')
    ap.add_argument('--datasets',nargs='+',default=list(DATASET_ORDER),choices=DATASET_ORDER)
    ap.add_argument('--n-queries',type=int,default=2,help='Pilot 2, full 200, selected with frozen seed')
    ap.add_argument('--seed',type=int,default=20260818)
    ap.add_argument('--repetitions',type=int,default=1)
    ap.add_argument('--warmup',type=int,default=20,help='Number of real query/matching kernel warm-ups outside timing')
    ap.add_argument('--mode',choices=['subset_calibrated','original_full_cohort_replay'],default='subset_calibrated')
    ap.add_argument('--feature-mode',choices=['prebuilt','on_the_fly'],default='on_the_fly')
    ap.add_argument('--prepare-gallery',action='store_true',help='Resumable offline precompute of gallery local features before timed run')
    ap.add_argument('--prepare-only',action='store_true',help='Precompute gallery features and exit; no latency claims')
    ap.add_argument('--overwrite-run',action='store_true',help='DISABLED. Never remove prior evidence in persistent mode')
    ap.add_argument('--drive-backup-root',required=True,help='Dedicated PERSISTENT Drive folder; must differ from /content local output')
    ap.add_argument('--checkpoint-every',type=int,default=10,help='Number of fresh query records per immutable Drive chunk')
    ap.add_argument('--resume-partial',action='store_true',help='Reuse validated saved query jobs. Suppress batch-level E2E claims on fragmented runs.')
    ap.add_argument('--gallery-cache',default='',help='Persistent path for precomputed gallery SuperPoint features')
    ap.add_argument('--data-preflight-only',action='store_true',help='Validate datasets and archive identity BEFORE loading GPU models')
    ap.add_argument('--stage-local-io',action='store_true',help='Offline stage selected query image bytes and gallery feature tensors to Colab local storage')
    args=ap.parse_args(argv)
    if args.overwrite_run:raise SystemExit('Persistent edition forbids --overwrite-run. Pick a fresh run ID or --resume-partial.')
    if args.checkpoint_every<1 or args.checkpoint_every>20:raise SystemExit('--checkpoint-every must be 1..20')
    args.resume_partial=True  # checkpoint-resume is always enabled and independently audited
    if args.n_queries<1 or args.n_queries>200:raise SystemExit('Supported timing cohort: 1..200 historical sample positions')
    if args.prepare_only and not args.prepare_gallery:raise SystemExit('--prepare-only requires --prepare-gallery')
    root=Path(args.package_root).resolve();ws=Path(args.workspace).resolve();out=Path(args.output).resolve()
    bk=Path(args.drive_backup_root).resolve()
    if not str(bk).startswith('/content/drive/'):
        raise SystemExit('The backup root MUST be on mounted Google Drive (/content/drive/...)')
    if bk==out or str(out).startswith(str(bk)+os.sep) or str(bk).startswith(str(out)+os.sep):
        raise SystemExit('Local measured output and persistent Drive backup must be disjoint')
    if not str(out).startswith('/content/') or str(out).startswith('/content/drive/'):
        raise SystemExit('Run local measurement output under /content but NOT in Drive')
    if not ws.is_dir():raise FileNotFoundError('Missing original historical workspace: '+str(ws))
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'src'))
    from vprbudget.data import load_dataset
    manifest_dir=ws/'entropybudget_e2e_persistent_runs'/'dataset_manifests'
    # Validate all dataset identifiers first. Previously a 27,592-file
    # Nordland DriveFS readdir failed AFTER downloading 700 MB of checkpoints.
    _cpu_preflight(ws,args.datasets,args.n_queries,args.seed,manifest_dir)
    if args.data_preflight_only:return
    bk.mkdir(parents=True,exist_ok=True)
    ctx=_load_models_and_sources(root,ws)
    torch=ctx['torch'];cfg=ctx['cfg']
    master_cache=Path(args.gallery_cache).resolve() if args.gallery_cache else ws/'entropybudget_e2e_gallery_local_cache'
    cache_dir=(out/'_local_gallery_cache') if args.stage_local_io else master_cache
    stage_meta=_env_info(ws,cfg,ctx['bo_q'],ctx['boq_provenance']);stage_meta.update({'mode':args.mode,'feature_mode':args.feature_mode,
                       'workspace_type':'user-authorized historical Drive workspace',
                       'gallery_state':'offline resident FAISS IndexFlatL2; built from original BoQ descriptor shards',
                       'query_features':'fresh inside measured prepass/local path; never cached',
                       'raw_query_image_io':'LOCAL pre-staged, bytes transferred OFFLINE' if args.stage_local_io else 'mounted DriveFS',
                       'gallery_feature_io':'LOCAL pre-staged tensors' if args.stage_local_io else 'mounted DriveFS',
                       'decoding':'Included (PIL global, LightGlue local); gallery local image decoding included only for on_the_fly mode',
                       'cohort_size':args.n_queries,'subset_seed':args.seed,
                       'source_hashes':{str(p.relative_to(root)):_sha(p) for p in (root/'src/Q1_Budget_Safe_Allocator_v1.py',root/'src/Q1_External_Confirmatory_Runner_v3.py',Path(__file__).resolve(),root/'src/persistent_checkpoint.py',root/'src/vprbudget/data.py')},
                       'candidate_size':'Top-100 retrieval; candidate order verified against archived retrieval where possible',
                       'limitations':'New subset calibration is not the original full-cohort frozen allocation. Original_full_cohort_replay excludes calibration cost.'})
    old_env_path=out/'environment.json'
    durable_backup_env=bk/'environment.json'
    if durable_backup_env.is_file():
        if old_env_path.is_file() and old_env_path.read_bytes()!=durable_backup_env.read_bytes():
            raise RuntimeError('Local and Drive environment.json differ: refuse mixing')
        if not old_env_path.is_file():
            old_env_path.write_bytes(durable_backup_env.read_bytes())
    if args.resume_partial and old_env_path.exists():
        old_env=json.loads(old_env_path.read_text())
        compared=['device','torch','cuda','gpu_driver','lightglue_commit','mode','feature_mode','cohort_size','subset_seed','source_hashes','checkpoint_hashes','raw_query_image_io','gallery_feature_io']
        changed=[k for k in compared if old_env.get(k)!=stage_meta.get(k)]
        # Repo acquisition method changes from 'git_clone' to 'persistent_cache'
        # after the first run; compare immutable source fingerprints, not route.
        for field in ('repository_commit','hubconf_sha256'):
            if (old_env.get('boq_source_recovery') or {}).get(field)!=(stage_meta.get('boq_source_recovery') or {}).get(field):
                changed.append('boq_source_recovery.'+field)
        if changed:raise RuntimeError('Cannot resume mismatched hardware/source/config: '+str(changed))
    if not old_env_path.is_file():
        _atomic_json(out/'environment.json',stage_meta)
    durable.persist_artifact(out,bk,'environment.json')
    original_positions=_frozen_subsets(ws,args.datasets,args.n_queries,args.seed)
    summaries=[]
    for dsname in args.datasets:
        ds=load_dataset(dsname,ws/'data_external_q1_v3',manifest_dir=manifest_dir)
        ret_path=ws/'q1_external_confirmatory_v3'/'boq-dinov2'/'retrieval'/f'{dsname}.npz'
        if not ret_path.is_file():raise FileNotFoundError('Frozen external original retriever shortlist absent: '+str(ret_path))
        with np.load(ret_path) as r: ret={k:r[k].copy() for k in r.files}
        if ret['candidates'].shape[1]!=100 or len(ret['query_indices'])!=ds.n_queries:
            raise RuntimeError(f'Dataset retrieval mismatch {dsname}: {ds.n_queries} and {ret["candidates"].shape}')
        select=original_positions[dsname]
        if args.stage_local_io:
            query_stage=_stage_local_queries(ds,ret,select,out)
            stage_meta['query_image_set_sha256']=query_stage['content_manifest_sha256']
            _atomic_json(out/'offline_raw_image_stage'/f'{dsname}.json',query_stage)
            durable.persist_first_provenance(out,bk,Path('offline_raw_image_stage')/f'{dsname}.json')
        shards=_shard_dirs(ws,dsname)
        t=time.perf_counter();idx,count=_make_faiss_index(shards,ds.n_database)
        live_cand,live_report=_live_candidate_preflight(ctx,ds,ret,select,idx,out,bk)
        if args.feature_mode=='prebuilt' or args.prepare_gallery:
            prep=_prepare_features(ctx['extractor'],ds,ret,select,cfg,master_cache,ctx['device'],ctx['lg'],live_candidate_positions=live_cand)
            if args.stage_local_io:
                prep['local_gallery_mirror']=_mirror_local_gallery_cache(ds,live_cand,cfg,master_cache,cache_dir)
            prep['candidate_source']='LIVE BoQ query retrieval; not historical candidate union'
            prep['preflight_candidate_matrix_sha256']=live_report['candidate_matrix_sha256']
            _atomic_json(out/'preparation'/f'{dsname}.json',prep)
            durable.persist_first_provenance(out,bk,Path('preparation')/f'{dsname}.json')
            _assert_cache_coverage(ds,cache_dir,cfg,live_cand,'OFFLINE PREP COMPLETE')
        if args.prepare_only:
            del idx
            continue
        _atomic_json(out/'index_preparation'/f'{dsname}.json',{'gallery':ds.n_database,
                    'shards':[{'name':p.name,'size_bytes':p.stat().st_size} for p in shards],
                    'shard_counts':count,'retrieval_archive_sha256':_sha(ret_path),
                    'gallery_shard_integrity':'file sizes recorded; full gallery byte hashes not verified',
                    'index_build_ms':(time.perf_counter()-t)*1000,'index_kind':'FAISS IndexFlatL2',
                    'timing_boundary':'OFFLINE; not included in E2E query/batch latency'})
        durable.persist_first_provenance(out,bk,Path('index_preparation')/f'{dsname}.json')
        _warm_query_storage(ds,ret,select)
        warm=_hardware_warmup(ctx,ds,ret,select,idx,cfg,cache_dir,args.feature_mode,args.warmup)
        _atomic_json(out/'preparation'/f'{dsname}_warmup.json',warm)
        durable.persist_first_provenance(out,bk,Path('preparation')/f'{dsname}_warmup.json')
        print('Dataset',dsname,'queries=',len(select),'gallery=',ds.n_database,'mode=',args.mode,'warmup=',warm['warmup_queries'],flush=True)
        for rep in range(args.repetitions):
            order=list(POLICIES)
            random.Random(args.seed+rep+sum(map(ord,dsname))).shuffle(order)
            for method in order:
                completed=bk/'summaries'/f'{dsname}_rep{rep:02d}_{method}.json'
                if completed.is_file():
                    # Full completed method is already durable. No replay/mixing required.
                    summary=json.loads(completed.read_text())
                    csv_p=bk/'records'/dsname/f'rep_{rep:02d}'/method/'query_timings.csv'
                    if summary.get('n')!=len(select) or summary.get('method')!=method or not csv_p.is_file():
                        raise RuntimeError('Completed Drive summary/CSV disagrees with this run')
                    print(f'>> SKIPPING COMPLETE DURABLE {dsname} rep={rep} {method}',flush=True)
                    summaries.append(summary)
                    continue
                _warm_query_storage(ds,ret,select)
                print(f'>> {dsname}, repetition {rep}, method {method}',flush=True)
                summary=_run_one(ctx,ds,ret,select,idx,cfg,cache_dir,out,method,rep,args,stage_meta)
                summaries.append(summary)
            gc.collect()
        del idx
    if args.prepare_only:
        _atomic_json(out/'PREPARATION_COMPLETE.json',{'status':'offline_gallery_preparation_only','datasets':args.datasets,'utc':_now()})
        durable.persist_first_provenance(out,bk,'PREPARATION_COMPLETE.json')
        return
    _atomic_json(out/'all_summary.json',summaries)
    import pandas as pd
    pd.json_normalize(summaries).to_csv(out/'e2e_summary.csv',index=False)
    allrows=[]
    for item in summaries:
        f=bk/'records'/item['dataset']/f"rep_{int(item['repetition']):02d}"/item['method']/'query_timings.csv'
        if not f.is_file():raise FileNotFoundError('Completed policy has no durable timing CSV: '+str(f))
        part=pd.read_csv(f)
        if len(part)!=args.n_queries:raise RuntimeError('Incorrect durable CSV query count: '+str(f))
        allrows.append(part)
    all_raw=pd.concat(allrows,ignore_index=True)
    all_raw.to_csv(out/'e2e_raw_query_timings.csv',index=False)
    per_q=all_raw.groupby(['dataset','repetition','query_index'])['live_top100_identity_sha256'].nunique()
    discrepant=int((per_q>1).sum())
    _atomic_json(out/'retrieval_identity_across_methods.json',{'compared_query_triplets':len(per_q),
                    'different_top100_between_policies':discrepant,
                    'all_identical':discrepant==0,
                    'warning':'If any disagreement, same-candidate-policy latency comparison requires investigation.'})
    # Negative or divergent retrieval replay is retained; no silent calibration replacement.
    partial=any(x.get('reused_previous_query_timings',0)>0 for x in summaries)
    _atomic_json(out/'RESULTS_STATUS.json',{'status':'MEASURED_QUERY_LEVEL_WITH_DRIVE_CHECKPOINTS_NO_CONTINUOUS_BATCH_TIMING','utc':_now(),
       'policies':list(POLICIES),'datasets':list(args.datasets),'repetitions':args.repetitions,
       'mode':args.mode,'n_queries_per_dataset':args.n_queries,
       'complete_cohort_benchmark':args.n_queries==200 and set(args.datasets)==set(DATASET_ORDER),
       'cross_policy_retrieval_consistent':discrepant==0,
       'checkpointed_to_drive':True,'drive_backup_root':str(bk),
       'raw_query_image_io':stage_meta['raw_query_image_io'],
       'gallery_feature_io':stage_meta['gallery_feature_io'],
       'continuous_batch_latency_valid':False,
       'interpretation':'Subset-calibrated method prepass is measured; per-query fresh inference stages are measured. Drive checkpoint pauses make UNINTERRUPTED batch time nonreportable.'})
    for name in ('all_summary.json','e2e_summary.csv','e2e_raw_query_timings.csv','retrieval_identity_across_methods.json','RESULTS_STATUS.json'):
        durable.persist_artifact(out,bk,name)
    print('FINISHED persistent hardware run; Drive evidence:',bk,flush=True)

if __name__=='__main__':main()

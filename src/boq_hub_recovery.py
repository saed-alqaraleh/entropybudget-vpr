"""Network-resilient OFFLINE-SOURCE loading of the official BoQ-DINOv2 hub model.

The original study's BoQ model, training/checkpoint and transforms are NOT
changed. The only difference is using PyTorch's documented source='local' mode
on a persistent copy of the *official* amaralibey/Bag-of-Queries repository.
That bypasses torch.hub's fragile GitHub /tree/main branch-discovery GET.

Scientific caveat: the recorded study pipeline did not freeze the BoQ commit.
Record repo_commit (or hubconf checksum for archive fallback), checkpoint
hashes, and actual retrieved top-100 identity agreement before claiming a
faithful image-level replay.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

OFFICIAL_REPO = 'https://github.com/amaralibey/Bag-of-Queries.git'
OFFICIAL_ARCHIVE = 'https://codeload.github.com/amaralibey/Bag-of-Queries/zip/refs/heads/main'
REPO_ENV = 'ENTROPYBUDGET_BOQ_REPO'


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1 << 20),b''):
            h.update(chunk)
    return h.hexdigest()


def _valid_repo(path):
    path=Path(path)
    return (path/'hubconf.py').is_file() and (path/'src').is_dir()


def _git_head(path):
    try:
        return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True,timeout=12).strip()
    except (FileNotFoundError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return None


def _recover_by_clone(target):
    temp=target.with_name(target.name+'.clone_incomplete')
    shutil.rmtree(temp,ignore_errors=True)
    cmd=['git','-c','http.lowSpeedLimit=1000','-c','http.lowSpeedTime=25',
         'clone','--depth','1','--single-branch','--branch','main',OFFICIAL_REPO,str(temp)]
    outcome=subprocess.run(cmd,text=True,capture_output=True,timeout=160,check=False)
    if outcome.returncode != 0 or not _valid_repo(temp):
        shutil.rmtree(temp,ignore_errors=True)
        raise RuntimeError(f'Official git clone failed (exit {outcome.returncode}): {outcome.stderr[-1500:]}')
    os.replace(temp,target)
    return 'git_clone_official_main'


def _recover_by_archive(target):
    # codeload is the official upstream archive endpoint, not a third-party mirror.
    with tempfile.TemporaryDirectory(prefix='entropybudget_boq_') as t:
        tmp=Path(t)
        raw=tmp/'upstream.zip'
        with urllib.request.urlopen(OFFICIAL_ARCHIVE,timeout=90) as response, raw.open('wb') as sink:
            shutil.copyfileobj(response,sink)
        with zipfile.ZipFile(raw) as z:
            bad=z.testzip()
            if bad:raise RuntimeError('Official GitHub archive corrupt: '+bad)
            names=[n for n in z.namelist() if n and not n.endswith('/')]
            roots={Path(n).parts[0] for n in names}
            if len(roots)!=1 or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
                raise RuntimeError('Unexpected/unsafe official GitHub archive layout')
            z.extractall(tmp/'expanded')
            source=tmp/'expanded'/next(iter(roots))
            if not _valid_repo(source):raise RuntimeError('Official archive lacks hubconf.py or src')
            temp=target.with_name(target.name+'.archive_incomplete')
            shutil.rmtree(temp,ignore_errors=True)
            shutil.copytree(source,temp)
            os.replace(temp,target)
    return 'github_codeload_official_main'


def prepare_official_boq_source(workspace, max_attempts=3):
    """Return (local_repo_path, immutable-run provenance). No model inference."""
    import torch
    work=Path(workspace).expanduser().resolve()
    candidates=[]
    if os.environ.get(REPO_ENV):
        candidates.append((Path(os.environ[REPO_ENV]).expanduser(), 'author_provided_official_checkout'))
    candidates.extend([
        (work/'entropybudget_e2e_model_repos'/'Bag-of-Queries','persistent_official_checkout'),
        (Path(torch.hub.get_dir())/'amaralibey_bag-of-queries_main','torch_hub_cache'),
        (Path(torch.hub.get_dir())/'amaralibey_Bag-of-Queries_main','torch_hub_cache'),
        (Path('/content/Bag-of-Queries'),'existing_colab_checkout'),
    ])
    chosen=None;origin=None
    for path,reason in candidates:
        if _valid_repo(path):chosen,origin=path,reason;break
    if chosen is None:
        target=work/'entropybudget_e2e_model_repos'/'Bag-of-Queries'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise RuntimeError(f'Existing non-repository at {target}. Move it aside; will NOT delete user data.')
        errors=[]
        for mode,fn in [('git',_recover_by_clone),('official_archive',_recover_by_archive)]:
            for n in range(max_attempts):
                try:
                    print(f'RECOVERING OFFICIAL BoQ REPO: {mode}, attempt {n+1}/{max_attempts}',flush=True)
                    origin=fn(target)
                    chosen=target
                    break
                except Exception as e:
                    errors.append(f'{mode} attempt {n+1}: {type(e).__name__}: {e}')
                    if n+1<max_attempts:time.sleep(min(2**n,8))
            if chosen is not None:break
        if chosen is None:
            raise RuntimeError('Could not obtain the unchanged official BoQ source. This is a network '
              'blocker, not model failure. Retry when GitHub responds, or provide an authentic '
              f'local official checkout using {REPO_ENV}. Errors: '+ ' | '.join(errors[-4:]))
    chosen=Path(chosen).resolve()
    commit=_git_head(chosen)
    meta={'repository':'amaralibey/Bag-of-Queries','entry_point':'get_trained_boq',
          'backbone_name':'dinov2','output_dim':12288,'source_type':'torch_hub_local',
          'checkout_origin':origin,'repository_path':str(chosen),
          'repository_commit':commit,'hubconf_sha256':_sha256(chosen/'hubconf.py'),
          'historical_commit_proven':False,
          'warning':'Compare live retrieved identities with the original frozen gallery and query retrieval before calling this exact historical model replication.'}
    print('OFFICIAL BoQ SOURCE READY:',meta['checkout_origin'], 'commit',commit or '(archive/no git)',flush=True)
    return chosen,meta


def load_historical_boq_wrapper(historical_runner,device,workspace):
    """Same official Hub entry point and same original DescriptorWrapper as history."""
    import torch
    local_repo,provenance=prepare_official_boq_source(workspace)
    original=torch.hub.load_state_dict_from_url
    original_hub_load=torch.hub.load
    def retry_official_weights(*a,**kw):
        # Preserves URL, model parameters, checkpoint path, map_location, etc.
        for attempt in range(3):
            try:return original(*a,**kw)
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ConnectionError,OSError) as exc:
                if attempt==2:raise
                print(f'Official checkpoint network issue ({type(exc).__name__}). Retrying exact URL.',flush=True)
                time.sleep(min(2**attempt,8))
    def route_nested_official_hub(repo_or_dir,entrypoint,*args,**kwargs):
        # DINOv2 is a dependency of the official BoQ hubconf in some upstream
        # revisions. An explicit :main ref avoids its own /tree/main probe.
        if repo_or_dir=='facebookresearch/dinov2' and kwargs.get('source','github')=='github':
            kw=dict(kwargs)
            kw['skip_validation']=True
            return original_hub_load('facebookresearch/dinov2:main',entrypoint,*args,**kw)
        return original_hub_load(repo_or_dir,entrypoint,*args,**kwargs)
    torch.hub.load_state_dict_from_url=retry_official_weights
    torch.hub.load=route_nested_official_hub
    try:
        # Exact same BoQ class, architecture and pretrained weight URL as
        # hist.load_official_retriever(). Only the source routing differs.
        model=torch.hub.load(str(local_repo),'get_trained_boq',
              backbone_name='dinov2',output_dim=12288,source='local',trust_repo=True)
    finally:
        torch.hub.load_state_dict_from_url=original
        torch.hub.load=original_hub_load
    model=model.eval().to(device)
    source={'repository':'amaralibey/bag-of-queries',
            'entry_point':"get_trained_boq(backbone_name='dinov2', output_dim=12288)",
            'paper':'BoQ: A Place is Worth a Bag of Learnable Queries (CVPR 2024)',
            'source_routing':'original official hubconf via torch.hub source=local'}
    return historical_runner.DescriptorWrapper(model),source,provenance

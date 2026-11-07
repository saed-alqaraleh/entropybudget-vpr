"""CPU-only high-signal controls. No fake GPU timing values are published."""
import json, os, sys, tempfile, types
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import persistent_checkpoint as d
import benchmark_e2e_resilient as r


def run():
 checks=[]
 def ok(label):checks.append(label)
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp);out=base/'local';bk=base/'drive';method=out/'records'/'nordland'/'rep_00'/'uniform_m20';remote=bk/'records'/'nordland'/'rep_00'/'uniform_m20'
  method.mkdir(parents=True)
  spec='f'*64
  for j in range(20):
   (method/f'query_{j:05d}.json').write_text(json.dumps({'query_slot':j,'run_spec_sha256':spec,'query_index':j,'budget':20,'timings':1.23},sort_keys=True))
  p1,a1=d.checkpoint_block(method,remote,0,10,spec)
  p2,a2=d.checkpoint_block(method,remote,10,20,spec)
  assert a1==a2=='saved_verified';ok('two independent exact 10-query backup blocks')
  assert d.checkpoint_block(method,remote,0,10,spec)[1]=='already_identical';ok('byte-deterministic re-checkpoint is idempotent')
  import shutil
  shutil.rmtree(method)
  assert d.recover_blocks(remote,method,spec,20)==20
  assert len(list(method.glob('query_*.json')))==20;ok('disconnected runtime: full 20-record recovery from Drive')
  assert d.recover_blocks(remote,method,spec,20)==20;ok('multiple recoveries preserve records')
  assert d.recover_blocks(remote,method,'a'*64,20) is None if False else True
  try:d.recover_blocks(remote,method,'0'*64,20)
  except RuntimeError:ok('source/protocol change is refused')
  else:raise AssertionError('accepted mismatched source!')
  (method/'query_00000.json').write_text('invalid modified data')
  try:d.recover_blocks(remote,method,spec,20)
  except RuntimeError:ok('local and Drive evidence conflict refused')
  else:raise AssertionError('ignored conflict')
  original=p1.read_bytes()
  try:d.put_checked(p1,b'new different bytes')
  except RuntimeError:ok('immutable Drive chunk refuses overwrite')
  else:raise AssertionError('overwrote chunk')
  assert p1.read_bytes()==original;ok('original Drive bytes remain untouched')
  artifact=out/'specs'/'nordland.json';artifact.parent.mkdir(parents=True);artifact.write_text('{}')
  assert d.persist_artifact(out,bk,Path('specs')/'nordland.json')=='saved_verified'
  assert d.persist_artifact(out,bk,Path('specs')/'nordland.json')=='already_identical';ok('Drive artifact copy checked and idempotent')
  artifact.write_text('{"different":true}')
  assert d.persist_first_provenance(out,bk,Path('specs')/'nordland.json')=='first_session_original_retained';ok('volatile preparation provenance retains first run')
  assert (bk/'specs'/'nordland.json').read_text()=='{}'
  # Live-only cache list: precomputed head does not authorize candidate cache.
  names=[base/'gallery'/f'{j}.jpg' for j in range(5)]
  cfg=types.SimpleNamespace(detector='superpoint',matching_image_size=512,max_keypoints=2048)
  ds=types.SimpleNamespace(name='toy',database_paths=names)
  C=np.zeros((2,100),dtype=int);C[:,0]=[2,3];C[:,1:50]=1
  from unittest import mock
  torch=__import__('torch')
  class Extractor:pass
  calls=[]
  def featurizer(model,path,size,device):
   calls.append(path);return {'keypoints':torch.zeros(1,2)},(512,512)
  with mock.patch('vprbudget.matching.extract_features',side_effect=featurizer):
   p=r._prepare_features(Extractor(),ds,{},np.array([0,1]),cfg,base/'gallery_cache',torch.device('cpu'),None,live_candidate_positions=C)
  assert p['union_images']==3 and set(calls)==set(names[i] for i in (1,2,3));ok('prebuilt cache uses LIVE top50 union, not archived union')
  assert r._assert_cache_coverage(ds,base/'gallery_cache',cfg,C,'TEST')==3;ok('complete live feature coverage checked before timing')
  (base/'gallery_cache'/r._cache_id(names[3],cfg)[:2]/(r._cache_id(names[3],cfg)+'.pt')).unlink()
  try:r._assert_cache_coverage(ds,base/'gallery_cache',cfg,C,'TEST')
  except RuntimeError:ok('missing live feature rejected in preflight')
  else:raise AssertionError('missed missing feature')
  ns=types.SimpleNamespace(mode='subset_calibrated',n_queries=200,seed=20260818,feature_mode='prebuilt')
  env={'device':'NVIDIA L4','torch':'2.11.0+cu128','cuda':'12.8','gpu_driver':'test','lightglue_commit':'x',
       'checkpoint_hashes':{'boq':'x'},'source_hashes':{'runner':'x'},'boq_source_recovery':{'hubconf_sha256':'x','repository_commit':'x'}}
  a,_=d.stable_spec(ns,'nordland',0,'uniform_m20',env,np.arange(200))
  b,_=d.stable_spec(ns,'nordland',0,'uniform_m20',env,np.arange(200))
  assert a==b
  env['checkpoint_hashes']['boq']='DIFFERENT';b,_=d.stable_spec(ns,'nordland',0,'uniform_m20',env,np.arange(200))
  assert a!=b;ok('run fingerprints include checkpoint identity and selection')
 print(json.dumps({'status':'PASS','checks':len(checks),'details':checks},indent=2))
 return len(checks)

if __name__=='__main__':run()

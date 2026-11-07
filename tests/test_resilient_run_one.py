"""Synthetic CPU process-disconnect test: results are NOT empirical timings."""
import json,sys,tempfile,types,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import benchmark_e2e_resilient as r
from vprbudget.config import Config
class FakeHistory:
 def official_input_transform(self):return None
 def retrieval_entropy(self,dist,T):return np.linspace(.1,.9,len(dist))
 def stable_top_fraction(self,ent,frac,largest=True):
  mask=np.zeros(len(ent),dtype=bool);mask[-int(round(len(ent)*frac)):]=True;return mask
class FakeIndex:
 def search(self,desc,k):
  return np.linspace(.1,3,k,dtype=np.float32)[None,:]**2,np.arange(k,dtype=np.int64)[None,:]

def descriptor(*args):return np.zeros(12288,np.float32)

def local(*args):
 return 0,dict(query_local_decode_extract_ms=.1,gallery_local_access_ms=.2,
               lightglue_matching_ms=.3,magsac_spatial_ms=.4,fixed_scoring_ms=.5)

def run():
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp);out=base/'ephemeral';bk=base/'mounted_drive'
  N=20
  ds=types.SimpleNamespace(name='nordland',query_paths=[Path(f'q{i}.jpg') for i in range(N)],database_paths=[Path('x.jpg')]*100)
  ret={'query_indices':np.arange(N),'candidates':np.tile(np.arange(100),(N,1)),
       'descriptor_distances':np.tile(np.linspace(0,1,100),(N,1))}
  cfg=Config()
  ctx={'torch':__import__('torch'),'hist':FakeHistory(),'bo_q':object(),'device':'cpu','cfg':cfg}
  args=types.SimpleNamespace(mode='subset_calibrated',feature_mode='on_the_fly',overwrite_run=False,
      resume_partial=True,drive_backup_root=str(bk),checkpoint_every=5,n_queries=N,seed=20260818)
  report={'device':'NVIDIA L4','torch':'2.11.0+cu128','cuda':'12.8','gpu_driver':'test','lightglue_commit':'test',
          'checkpoint_hashes':{'boq':'x'},'source_hashes':{'benchmark':'test'},
          'boq_source_recovery':{'hubconf_sha256':'test','repository_commit':'test'}}
  interrupted={'calls':0}
  def crash_after_12(*a):
   interrupted['calls']+=1
   if interrupted['calls']>12:raise InterruptedError('simulated process crash after 12 completed queries')
   return local(*a)
  with patch.object(r,'_model_descriptor',descriptor),patch.object(r,'_query_local_pipeline',crash_after_12):
   try:r._run_one(ctx,ds,ret,np.arange(N),FakeIndex(),cfg,base/'cache',out,'uniform_m20',0,args,report)
   except InterruptedError:pass
   else:raise AssertionError('Expected simulated disconnect')
  remote=bk/'records'/'nordland'/'rep_00'/'uniform_m20'/'chunks'
  assert len(list(remote.glob('*.zip')))==2,'Must have 10 permanent records at crash'
  assert len(list((out/'records'/'nordland'/'rep_00'/'uniform_m20').glob('query_*.json')))==12
  # A *new* ephemeral filesystem after a Colab runtime replacement.
  shutil.rmtree(out)
  invoked={'n':0}
  def resumed_local(*a):
   invoked['n']+=1
   return local(*a)
  with patch.object(r,'_model_descriptor',descriptor),patch.object(r,'_query_local_pipeline',resumed_local):
   summary=r._run_one(ctx,ds,ret,np.arange(N),FakeIndex(),cfg,base/'cache',out,'uniform_m20',0,args,report)
  assert invoked['n']==10,'Recomputed queries 10..19, not lost original 10'
  assert summary['reused_previous_query_timings']==10
  assert summary['complete_batch_wall_ms'] is None and not summary['cross_fragment_e2e_valid']
  assert summary['batch_timing_status'].startswith('NOT_REPORTABLE')
  assert len(list(remote.glob('*.zip')))==4
  assert (bk/'records'/'nordland'/'rep_00'/'uniform_m20'/'query_timings.csv').exists()
  assert (bk/'summaries'/'nordland_rep00_uniform_m20.json').exists()
  print(json.dumps({'status':'PASS','scenario':'simulated disconnect after query 12',
     'drive_preserved_before_disconnect':10,'resumed_from_drive':10,
     'newly_computed_after_disconnect':10,'final_rows':20,
     'uninterrupted_batch_claim':False},indent=2))

if __name__=='__main__':run()

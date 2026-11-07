#!/usr/bin/env python3
"""Fast real-evidence CPU smoke; not an image-level GPU replay."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vprbudget.config import load_config
from vprbudget.scoring import global_score,global_reliability
from vprbudget.groundtruth import positives,get_spec
import budget_safe_allocator_v1 as safe
cfg=load_config(str(ROOT/'configs/run02_sf_partial_v2.json'))
core=pd.read_csv(ROOT/'validation_inputs/megaloc_per_query_core_for_ci.csv')
checks=0; ds_rows=[]
for ds in cfg.datasets:
 with np.load(ROOT/f'validation_inputs/retrieval/{ds}.npz') as r,np.load(ROOT/f'validation_inputs/scoring/{ds}.npz') as s:
  dist=r['descriptor_distances']; G=s['scores'][:,0,:]
  n=s['inlier_count'];mc=s['match_count'];sp=s['spatial'];cg=s['global_reliability']
  pos=positives(r['geo_distances'],get_spec(ds,cfg.dataset_gt_overrides))
  archived=core[core.dataset==ds].sort_values('i')
  indices=np.unique(np.linspace(0,len(G)-1,min(4,len(G))).astype(int))
  for ix in indices:
   np.testing.assert_allclose(global_score(dist[ix],cfg.beta_global),G[ix],atol=1e-6,rtol=1e-6)
   assert abs(global_reliability(dist[ix],cfg.softmax_temperature,cfg.global_reliability_weights)-cg[ix])<1e-6
   for depth in (0,20,50,100):
    j=int(safe.budget_safe_top1(G[ix:ix+1],n[ix:ix+1],mc[ix:ix+1],sp[ix:ix+1],cg[ix:ix+1],depth,cfg)[0])
    assert int(pos[ix,j])==int(archived.iloc[ix][f'c{depth}']), (ds,ix,depth)
    checks+=1
  ds_rows.append(dict(dataset=ds,rows=len(G),sampled_queries=len(indices),depths=[0,20,50,100]))
print(json.dumps({'status':'PASS','scope':'authentic stored-score subset smoke (not GPU/image inference)','correctness_decisions':checks,'datasets':ds_rows},indent=2))

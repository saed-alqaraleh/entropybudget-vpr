#!/usr/bin/env python3
"""Self-contained CPU rerun of the two 200-query candidate/mask replay checks."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'retrieval_replay_inputs'
summary=json.loads((root/'RESULTS.json').read_text())
assert summary['cohort_size']==200
for ds in ('nordland','baidu'):
 with np.load(root/f'{ds}_200_query_replay_inputs.npz') as z:
  archived=z['archive_candidates'];live=z['live_candidates'];dist=z['archive_distances'];gpu=z['live_selected'].astype(bool)
  assert archived.shape==live.shape==(200,100) and dist.shape==(200,100)
  assert np.all(archived[:,0]==live[:,0])
  ordered100=int(np.sum(np.all(archived==live,axis=1)))
  sets100=np.array([len(set(map(int,a))&set(map(int,b))) for a,b in zip(archived,live)])
  sets50=np.array([len(set(map(int,a[:50]))&set(map(int,b[:50]))) for a,b in zip(archived,live)])
  assert gpu.sum()==80
  # Ascending normalized distance, entropy T=0.1; archive float32 is sufficient for these cohorts.
  dd=dist.astype(np.float64);lo=dd.min(axis=1,keepdims=True);hi=dd.max(axis=1,keepdims=True);span=hi-lo
  x=np.divide(dd-lo,span,out=np.zeros_like(dd),where=span>1e-12);x=np.where(span<=1e-12,.5,x)
  logits=-x/.1;logits-=logits.max(axis=1,keepdims=True)
  pr=np.exp(logits);pr/=pr.sum(axis=1,keepdims=True)
  entropy=-(pr*np.log(np.maximum(pr,1e-12))).sum(axis=1)/np.log(100)
  selected=np.zeros(200,dtype=bool);selected[np.argsort(entropy,kind='stable')[::-1][:80]]=True
  assert np.array_equal(selected,gpu),(ds,'allocation mask changed')
  s=summary['datasets'][ds]
  assert ordered100==s['exact_full_top100_ordered']
  assert int(np.sum(sets100==100))==s['exact_full_top100_set']
  assert np.isclose(np.mean(sets100/100),s['mean_full_top100_overlap'])
  assert int(np.sum(sets50[gpu]==50))==s['top50_set_exact_on_paid_queries']
  assert int(np.sum(100-sets100))==s['total_archive_top100_membership_changes']
  print(ds,'PASS: 200 top1 identities; top100 ordered',ordered100,'set',int(np.sum(sets100==100)), 'mean overlap',float(np.mean(sets100)/100), 'selected-mask agreement 200/200')
print('PASS: compact saved-source retrieval and same-allocation checks complete; no GPU access')

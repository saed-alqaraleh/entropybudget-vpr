#!/usr/bin/env python3
"""Reconstruct A100 descriptive table from recorded individual query stage timings.
NOT a continuous image-to-ranking timer; checkpoint pauses invalidate batch timing.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]/'profiling/a100'));args=p.parse_args();root=Path(args.root)
rows=[]
for dataset in ('nordland','baidu'):
 src=root/dataset/'e2e_raw_query_timings.csv'
 with src.open(newline='') as f:data=list(csv.DictReader(f))
 summary=json.loads((root/dataset/'all_summary.json').read_text())
 for method in ('entropy_budget','uniform_m20','uniform_m50'):
  subset=sorted((x for x in data if x['method']==method),key=lambda x:int(x['query_slot']))
  assert len(subset)==200 and [int(x['query_slot']) for x in subset]==list(range(200))
  x=np.array([float(q['conditional_service_ms']) for q in subset]);st=np.array([float(q['global_image_descriptor_ms'])+float(q['top100_search_ms'])+float(q['local_path_ms']) for q in subset]);assert np.allclose(x,st,rtol=0,atol=1e-6)
  budgets=np.array([int(q['budget']) for q in subset])
  target=next(s for s in summary if s['method']==method)
  assert np.isclose(x.mean(),target['service_mean_ms'],atol=1e-6)
  assert np.isclose(np.median(x),target['service_median_ms'],atol=1e-6)
  assert np.isclose(np.percentile(x,95),target['service_p95_ms'],atol=1e-6)
  assert not target.get('complete_batch_wall_ms')
  rows.append([dataset,method,len(x),float(budgets.mean()),int(budgets.sum()),round(x.mean(),3),round(float(np.median(x)),3),round(float(np.percentile(x,95)),3)])
print('dataset,policy,n,mean_budget,verified_pairs,mean_stage_sum_ms,median_stage_sum_ms,p95_stage_sum_ms')
for row in rows:print(','.join(map(str,row)))
print('PASS: exactly 1200 query-policy rows; all six saved summaries reconstructed; uninterrupted batch time NOT_MEASURED')

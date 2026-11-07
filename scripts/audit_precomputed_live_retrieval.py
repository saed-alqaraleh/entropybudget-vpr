#!/usr/bin/env python3
"""CPU-only, read-only audit of archived BoQ retrieval versus V2 live candidate IDs.

Usage:
 python audit_archived_live_retrieval.py \
    --archived-nordland /path/to/nordland.npz \
    --archived-baidu /path/to/baidu.npz \
    --v2-return /path/to/EntropyBudget_Resilient_GPU_V2_Return.zip \
    --output /path/to/audit-output

Does not call CUDA, modify any experiment, or inspect verification labels.
"""
import argparse, csv, hashlib, io, json, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

SEED=20260818
COHORT=200
K=100
DEPTH=50
FRACTION=.40
DATASET_ORDER=[('st_lucia',1549),('nordland',27592),('baidu',3208)]
TOPS=(1,5,10,20,50,100)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):h.update(block)
    return h.hexdigest()

def entropy(d,T=.10):
    d=np.asarray(d,np.float64)
    lo=d.min(axis=1,keepdims=True);hi=d.max(axis=1,keepdims=True)
    span=hi-lo
    dn=np.divide(d-lo,span,out=np.zeros_like(d),where=span>1e-12)
    dn=np.where(span<=1e-12,.5,dn)
    z=-dn/T;z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    return -(p*np.log(np.clip(p,1e-12,1.0))).sum(axis=1)/np.log(d.shape[1])

def top_fraction(x,frac=.40):
    order=np.argsort(np.asarray(x,np.float64),kind='stable')[::-1]
    result=np.zeros(len(x),bool);result[order[:int(round(frac*len(x)))]]=True
    return result

def cohort_positions():
    rng=np.random.default_rng(SEED)
    return {name:np.sort(rng.choice(n,size=min(n,COHORT),replace=False)).astype(np.int64)
            for name,n in DATASET_ORDER}

def read_csv(z,ds):
    name=f'{ds}/records/{ds}/rep_00/entropy_budget/query_timings.csv'
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8'))))

def full_audit(archived, v2_return, output):
    output.mkdir(parents=True,exist_ok=True)
    pos=cohort_positions()
    final={"analysis":"CPU-only archived-versus-live retrieval comparison","seed":SEED,"cohort_size":COHORT,
           "top_k":K,"entropy_temperature":.10,"high_uncertainty_fraction":FRACTION,
           "selected_depth":DEPTH,"historical_archive_hashes":{},"datasets":{}}
    csv_rows=[]
    with zipfile.ZipFile(v2_return) as z:
        assert z.testzip() is None, 'V2 outer ZIP failed CRC'
        for ds in ('nordland','baidu'):
            path=archived[ds];digest=sha256(path)
            meta=json.loads(z.read(f'{ds}/index_preparation/{ds}.json'))
            assert digest==meta['retrieval_archive_sha256'],f'{ds}: original retrieval archive SHA-256 differs from V2 environment'
            final['historical_archive_hashes'][ds]=digest
            with np.load(path) as a:
                query_idx=a['query_indices'][pos[ds]]
                historical=a['candidates'][pos[ds]].astype(np.int64)
                dist=a['descriptor_distances'][pos[ds]].astype(np.float64)
            live=np.load(io.BytesIO(z.read(f'{ds}/offline_live_preflight/{ds}_candidate_ids.npy')))
            assert historical.shape==live.shape==(COHORT,K)
            assert np.all(query_idx>=0)
            rows=read_csv(z,ds)
            assert len(rows)==COHORT
            assert np.array_equal(np.array([int(r['query_position']) for r in rows]),pos[ds])
            assert np.array_equal(np.array([int(r['query_index']) for r in rows]),query_idx)
            assert np.array_equal(np.array([int(r['budget']) for r in rows]),np.where(np.array([int(r['budget']) for r in rows])==50,50,0))
            assert np.array_equal(np.array([str(r['original_top1_same']).lower()=='true' for r in rows]),historical[:,0]==live[:,0])
            assert np.array_equal(np.array([str(r['original_top100_same']).lower()=='true' for r in rows]),np.all(historical==live,axis=1))
            for j,r in enumerate(rows):
                assert hashlib.sha256(np.asarray(live[j],dtype=np.int64).tobytes()).hexdigest()==r['live_top100_identity_sha256']
                assert len(set(historical[j]))==K and len(set(live[j]))==K
            historical_allocation=top_fraction(entropy(dist))
            gpu_allocation=np.asarray([int(r['budget'])==50 for r in rows])
            assert historical_allocation.sum()==gpu_allocation.sum()==80
            metrics=[]
            adjacent_inversion_gaps=[]; moved_historical_gaps=[]
            for j,(h,l,d) in enumerate(zip(historical,live,dist)):
                row={"dataset":ds,"query_slot":j,"archive_position":int(pos[ds][j]),"query_index":int(query_idx[j]),
                     "archive_entropy":float(entropy(d[None,:])[0]),
                     "historical_selected":bool(historical_allocation[j]),"live_selected":bool(gpu_allocation[j]),
                     "same_allocation":bool(historical_allocation[j]==gpu_allocation[j])}
                for k in TOPS:
                    row[f'exact_ordered_top{k}']=bool(np.array_equal(h[:k],l[:k]))
                    intersect=len(set(map(int,h[:k]))&set(map(int,l[:k])))
                    row[f'common_top{k}']=intersect
                    row[f'set_exact_top{k}']=bool(intersect==k)
                row['first_different_rank_1based']=int(np.flatnonzero(h!=l)[0])+1 if np.any(h!=l) else None
                oldpos={int(x):int(i) for i,x in enumerate(h)}
                for k,x in enumerate(l):
                    i=oldpos.get(int(x))
                    if i is not None and k!=i:
                        moved_historical_gaps.append(abs(float(d[i]-d[k])))
                for k in range(len(l)-1):
                    x,y=map(int,l[k:k+2]); ix=oldpos.get(x);iy=oldpos.get(y)
                    if ix is not None and iy is not None and ix>iy:
                        adjacent_inversion_gaps.append(abs(float(d[ix]-d[iy])))
                row['archived_candidate_missing_from_live100']=K-row['common_top100']
                row['archived_candidate_missing_from_live50']=50-row['common_top50']
                metrics.append(row)
                csv_rows.append(row)
            doc={"query_count":COHORT,
                "exact_full_top100_ordered":sum(r['exact_ordered_top100'] for r in metrics),
                "exact_full_top100_set":sum(r['set_exact_top100'] for r in metrics),
                "top1_identity":sum(r['exact_ordered_top1'] for r in metrics),
                "same_allocation_queries":sum(r['same_allocation'] for r in metrics),
                "selected_queries":sum(r['live_selected'] for r in metrics),
                "historical_selected_queries":sum(r['historical_selected'] for r in metrics),
                "selected_intersection":sum(r['historical_selected'] and r['live_selected'] for r in metrics),
                "top50_set_exact_on_paid_queries":sum(r['set_exact_top50'] for r in metrics if r['live_selected']),
                "top50_ordered_exact_on_paid_queries":sum(r['exact_ordered_top50'] for r in metrics if r['live_selected']),
                "mean_full_top100_overlap":float(np.mean([r['common_top100']/100 for r in metrics])),
                "mean_top50_overlap":float(np.mean([r['common_top50']/50 for r in metrics])),
                "total_archive_top100_membership_changes":int(sum(r['archived_candidate_missing_from_live100'] for r in metrics)),
                "queries_with_top100_membership_change":int(sum(r['archived_candidate_missing_from_live100']>0 for r in metrics)),
                "rank_first_changed_median_1based":float(np.median([r['first_different_rank_1based'] for r in metrics if r['first_different_rank_1based'] is not None])),
                "adjacent_inversions_count":len(adjacent_inversion_gaps),
                "inverted_neighbor_archived_distance_gap_median":float(np.median(adjacent_inversion_gaps)),
                "inverted_neighbor_archived_distance_gap_max":float(max(adjacent_inversion_gaps)),
                "moved_rank_archived_distance_gap_max":float(max(moved_historical_gaps))}
            for k in TOPS:
                doc[f'exact_ordered_top{k}']=sum(r[f'exact_ordered_top{k}'] for r in metrics)
                doc[f'exact_set_top{k}']=sum(r[f'set_exact_top{k}'] for r in metrics)
                doc[f'mean_set_overlap_top{k}']=float(np.mean([r[f'common_top{k}']/k for r in metrics]))
            final['datasets'][ds]=doc
            # Retain 200-query traces, not duplicated full 27K/3K-query archives.
            np.savez_compressed(output/f'{ds}_200_query_replay_inputs.npz',archive_positions=pos[ds],query_indices=query_idx,
                                archive_candidates=historical,archive_distances=dist.astype(np.float32),
                                live_candidates=live.astype(np.int64),live_selected=gpu_allocation)
    (output/'RESULTS.json').write_text(json.dumps(final,indent=2)+'\n')
    with (output/'PER_QUERY_RETRIEVAL_DIFFERENCES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    return final

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archived-nordland',required=True,type=Path)
    p.add_argument('--archived-baidu',required=True,type=Path)
    p.add_argument('--v2-return',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    d=full_audit({'nordland':a.archived_nordland,'baidu':a.archived_baidu},a.v2_return,a.output)
    print(json.dumps(d['datasets'],indent=2))

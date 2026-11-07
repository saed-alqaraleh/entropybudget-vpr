#!/usr/bin/env python3
"""Audit actual historical MegaLoc retrieval+local evidence against frozen outcomes.

No re-training, synthetic local data, or external labels for selection. Run with CPU.
Outputs are explicitly cached-source verification, not a full inference rerun.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, math
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

DEPTHS = (0, 1, 2, 5, 10, 20, 50, 100)
AUDIT_DEPTHS = (0, 1, 2, 5, 10, 20, 50)
ORIG_DEPTHS = (0, 20, 50, 100)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def module(path, name):
    s=importlib.util.spec_from_file_location(name, path)
    a=importlib.util.module_from_spec(s)
    sys.modules[name]=a
    s.loader.exec_module(a)
    return a

def minmax(x):
    x=np.asarray(x,dtype=np.float64)
    d=x.max()-x.min()
    return np.ones_like(x)*.5 if d < 1e-12 else (x-x.min())/d

def softmax(x):
    x=np.asarray(x,dtype=np.float64)
    p=np.exp(x-x.max());return p/p.sum()

def entropy(p):
    return float(-(p*np.log(np.maximum(p,1e-12))).sum()/np.log(len(p))) if len(p)>1 else 0

def independent_scalar_top1(dist, G, n, mc, sp, m, cfg):
    """Independent scalar transcription of the manuscript's fixed appendix equations."""
    order=np.argsort(-np.asarray(G),kind='stable')
    if m<=1:return int(order[0])
    head=order[:m]
    n=n[head].astype(float);mc=mc[head].astype(float);sp=sp[head].astype(float);Gh=G[head].astype(float)
    rho=n/np.maximum(mc,1)
    L=.7*minmax(np.log1p(n))+.3*rho
    SL=L*(.5+.5*sp)
    pg=softmax(-dist/.1)
    ps=np.sort(pg)[::-1]
    gap=(dist[1]-dist[0])/(dist.max()-dist.min()) if dist.max()-dist.min()>1e-12 else 0
    cg=.25*(ps[0]+ps[0]-ps[1]+gap+1-entropy(pg))
    pl=softmax(SL/.1)
    j=int(np.argmax(SL))
    ssl=np.sort(SL)[::-1]
    dl=(ssl[0]-ssl[1])/(ssl[0]-ssl[-1]) if ssl[0]-ssl[-1]>1e-12 else 0
    cl=.2*(pl[j]+dl+minmax(np.log1p(n))[j]+rho[j]+1-entropy(pl))
    lam=np.clip(cg/(cg+cl+1e-8),.05,.95)
    vectors=(Gh,L,.7*Gh+.3*L,.7*Gh+.3*SL,lam*Gh+(1-lam)*L,lam*Gh+(1-lam)*SL)
    rrf=np.zeros(len(head)); tk=np.zeros(len(head))
    for vec in vectors:
        perm=np.argsort(-vec,kind='stable'); ranks=np.empty(len(head),int);ranks[perm]=np.arange(1,len(head)+1)
        rrf+=1/(60+ranks)
        for cutoff in (5,10):tk+=(ranks<=cutoff)*(1-(ranks-1)/cutoff)
    fused=.6*minmax(vectors[-1])+.2*minmax(rrf)+.2*minmax(tk)
    return int(head[int(np.argmax(fused))])

def verify_raw_geometry(shard, scoring, support):
    z=np.load(shard); s=np.load(scoring)
    a=np.zeros_like(z['inlier_count'],float)
    off=z['offsets']; qc=z['inlier_xy_query'];cc=z['inlier_xy_cand']
    assert np.array_equal(z['query_positions'],np.arange(len(off)))
    assert np.array_equal(z['inlier_count'],s['inlier_count'][:len(off)])
    assert np.array_equal(z['match_count'],s['match_count'][:len(off)])
    for row in range(len(off)):
        for col in range(off.shape[1]):
            start,n=map(int,off[row,col])
            assert 0<=start<=len(qc) and 0<=n<=len(qc)-start
            if n:
                bq,cq=support.spatial_coverage(qc[start:start+n],z['query_wh'][row],(4,4))
                bi,ci=support.spatial_coverage(cc[start:start+n],z['cand_wh'][row,col],(4,4))
                a[row,col]=support.spatial_distribution_score(bq,bi,cq,ci)
    d=np.abs(a-s['spatial'][:len(off)])
    return dict(raw_queries=len(off),raw_pairs=int(np.prod(off.shape[:2])),inliers=int(z['inlier_count'].sum()),
                spatial_max_abs_error=float(d.max()),spatial_mismatches_above_1e_6=int((d>1e-6).sum()),
                inlier_and_match_count_identity=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--package-root',default=str(Path(__file__).resolve().parents[1]))
    args=ap.parse_args();root=Path(args.package_root)
    sys.path.insert(0,str(root/'src'))
    from vprbudget.config import load_config
    from vprbudget.scoring import global_score,global_reliability
    from vprbudget.groundtruth import positives,get_spec
    safe=module(root/'src/budget_safe_allocator_v1.py','frozen_safe')
    cfg=load_config(str(root/'configs/run02_sf_partial_v2.json'))
    core=pd.read_csv(root/'validation_inputs/megaloc_per_query_core_for_ci.csv')
    results=[];summaries=[];conflicts=[];ref_checks=[];source_hashes={}
    for ds in cfg.datasets:
        ret_path=root/f'validation_inputs/retrieval/{ds}.npz';sc_path=root/f'validation_inputs/scoring/{ds}.npz'
        source_hashes[str(ret_path.relative_to(root))]=sha256(ret_path)
        source_hashes[str(sc_path.relative_to(root))]=sha256(sc_path)
        with np.load(ret_path) as r,np.load(sc_path) as s:
            dist=r['descriptor_distances'].astype(float);geo=r['geo_distances']
            G=s['scores'][:,0,:].astype(float);n=s['inlier_count'];mc=s['match_count'];sp=s['spatial'];cg=s['global_reliability']
            ori=core[core.dataset==ds].sort_values('i')
            assert len(ori)==len(G)==len(r['query_indices'])
            assert np.array_equal(ori['i'].to_numpy(),np.arange(len(G)))
            assert n.shape==mc.shape==sp.shape==G.shape and G.shape[1]==100
            test_idx=np.arange(min(20,len(G)))
            g_err=max(float(np.max(np.abs(global_score(dist[i],cfg.beta_global)-G[i]))) for i in range(len(G)))
            cg_err=max(float(abs(global_reliability(dist[i],cfg.softmax_temperature,cfg.global_reliability_weights)-cg[i])) for i in range(len(G)))
            assert g_err < 1e-6 and cg_err < 1e-6, (ds,g_err,cg_err)
            pos=positives(geo,get_spec(ds,cfg.dataset_gt_overrides))
            order=np.argsort(-G,axis=1,kind='stable')
            for m in DEPTHS:
                pred=safe.budget_safe_top1(G,n,mc,sp,cg,m,cfg)
                assert len(pred)==len(G)
                cor=pos[np.arange(len(G)),pred].astype(np.int8)
                n_mis=0
                if m in ORIG_DEPTHS:
                    n_mis=int(np.sum(cor!=ori[f'c{m}'].to_numpy(dtype=np.int8)))
                    if n_mis:
                        bad=np.where(cor!=ori[f'c{m}'].to_numpy(dtype=np.int8))[0]
                        conflicts.extend(dict(dataset=ds,query_position=int(i),depth=m,
                                              calculated=int(cor[i]),archived=int(ori[f'c{m}'].iloc[i])) for i in bad)
                if m in AUDIT_DEPTHS:
                    tail=order[test_idx,m:]
                    mutant=[]
                    for mode in ('zeros','reverse','adversarial'):
                        nn=n[test_idx].copy();mm=mc[test_idx].copy();ss=sp[test_idx].copy()
                        if m<100:
                            idx=np.arange(len(test_idx))[:,None]
                            if mode=='zeros':
                                nn[idx,tail]=0;mm[idx,tail]=0;ss[idx,tail]=0
                            elif mode=='reverse':
                                nn[idx,tail]=nn[idx,tail[:,::-1]]
                                mm[idx,tail]=mm[idx,tail[:,::-1]]
                                ss[idx,tail]=ss[idx,tail[:,::-1]]
                            else:
                                nn[idx,tail]=10**6;mm[idx,tail]=10**6;ss[idx,tail]=1.
                        alt=safe.budget_safe_top1(G[test_idx],nn,mm,ss,cg[test_idx],m,cfg)
                        mutant.append((mode,int(np.count_nonzero(pred[test_idx]!=alt))))
                    assert all(x[1]==0 for x in mutant), (ds,m,mutant)
                else:
                    mutant=[]
                for i in test_idx:
                    ref=independent_scalar_top1(dist[i],G[i],n[i],mc[i],sp[i],m,cfg)
                    ref_checks.append(dict(dataset=ds,query_position=int(i),budget=m,
                                           independent_top1=ref,historical_top1=int(pred[i]),agree=bool(ref==pred[i])))
                for i in range(len(G)):
                    if m in ORIG_DEPTHS:
                        results.append(dict(dataset=ds,query_position=i,query_index=int(r['query_indices'][i]),
                                            depth=m,global_candidate_index=int(pred[i]),candidate_identity=int(r['candidates'][i,pred[i]]),
                                            calculated_correct=int(cor[i]),archived_correct=int(ori[f'c{m}'].iloc[i])))
                summaries.append(dict(dataset=ds,depth=m,n=len(G),correct=int(cor.sum()),
                                      archived_correctness_disagreements=n_mis if m in ORIG_DEPTHS else '',
                                      tail_modes='|'.join(v[0] for v in mutant),
                                      tail_disagreements=sum(v[1] for v in mutant),
                                      global_score_max_abs_error=g_err,global_reliability_max_abs_error=cg_err))
            print(ds,'passed',len(G),'authentic records',flush=True)
    out=root/'validation_results';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/'decision_reproduction.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'budget_fidelity_summary.csv',index=False)
    pd.DataFrame(conflicts,columns=['dataset','query_position','depth','calculated','archived']).to_csv(out/'discrepancies.csv',index=False)
    pd.DataFrame(ref_checks).to_csv(out/'independent_equations_spotcheck.csv',index=False)
    raw=verify_raw_geometry(root/'validation_inputs/raw_matching/svox_shard_00000.npz',
                            root/'validation_inputs/scoring/svox.npz',
                            module(root/'src/vprbudget/scoring.py','raw_spatial_backend'))
    assert raw['spatial_mismatches_above_1e_6']==0
    assert not conflicts and all(t['agree'] for t in ref_checks)
    report=dict(status='PASS',query_rows=len(core),archived_budget_decisions=len(results),
                archived_correctness_disagreements=len(conflicts),independent_reference_checks=len(ref_checks),
                independent_reference_disagreements=sum(not x['agree'] for x in ref_checks),
                real_tail_invariance_policies=AUDIT_DEPTHS,tail_modes=['zeros','reverse','adversarial'],
                real_evidence_queries_for_tail_invariance_per_dataset=20,
                raw_matching_geometry=raw,hashes=source_hashes,
                limitations='This validates cached genuine matching evidence and scorer logic. It does not independently regenerate image features, LightGlue matches, or MAGSAC estimates on the original images.')
    (out/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('hashes','limitations')},indent=2))

if __name__=='__main__':main()

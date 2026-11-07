#!/usr/bin/env python3
"""Stored-query Phase A analysis; no model inference. Python >=3.10, numpy, matplotlib.
Group boundaries use round(k*N/5) and stable entropy ranking on original query positions.
All uncertainty intervals are percentile paired nonparametric bootstrap, resampling physical queries.
"""
import csv, json, hashlib, pathlib, shutil, math
from collections import Counter
import numpy as np

BASE=pathlib.Path(__file__).resolve().parents[1]
INP=BASE/'inputs'
OUT=BASE
SEED=20261105
B_GROUP=2500
B_CONTRAST=5000
FIELD_INT=['query_position','query_index','frozen_policy_budget','margin_policy_budget','retrieval_correct','uniform_m20_correct','uniform_m50_correct','frozen_policy_correct','margin_policy_correct']
FIELD_FLT=['entropy_uncertainty','top12_margin']
FIELDS=['retriever','dataset']+FIELD_INT+FIELD_FLT

def rows(path):
    with open(path,newline='',encoding='utf-8') as f:
        return [{k:(int(v) if k in FIELD_INT else float(v) if k in FIELD_FLT else v) for k,v in r.items()} for r in csv.DictReader(f)]

def write(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not items: raise ValueError('No output rows '+str(path))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0].keys()), extrasaction='raise'); w.writeheader(); w.writerows(items)

def sha(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()

def pct(v): return float(np.mean(v)*100)

def ci(a,seed,B=2500):
    """Exact categorical implementation of paired nonparametric query bootstrap.
    Sampling multinomial counts of observed row vectors is distribution-equivalent
    to resampling original query indices, and avoids giant (B x N) arrays.
    """
    x=np.asarray(a,dtype=np.float64)
    if x.ndim==1: x=x[:,None]
    n=len(x)
    if n==0: return [[float('nan')]*2 for _ in range(x.shape[1])]
    patterns,counts=np.unique(x,axis=0,return_counts=True)
    rng=np.random.default_rng(seed)
    totals=rng.multinomial(n,counts/n,size=B)@patterns
    bounds=np.quantile(totals/n*100,[0.025,0.975],axis=0)
    return [list(map(float,bounds[:,j])) for j in range(x.shape[1])]

def keyed(src):
    di={}
    for r in src:
        key=(r['retriever'],r['dataset'])
        di.setdefault(key,[]).append(r)
    return {key: sorted(v,key=lambda z:z['query_position']) for key,v in di.items()}

def prepare(z,key):
    n=len(z)
    assert len({r['query_position'] for r in z})==n
    assert len({r['query_index'] for r in z})==n
    u=np.array([r['entropy_uncertainty'] for r in z],dtype=np.float64)
    order=np.argsort(u,kind='stable')
    cuts=[round(k*n/5) for k in (1,2,3,4)]
    q=np.zeros(n,dtype=np.int8)
    for i,(start,end) in enumerate(zip([0]+cuts,cuts+[n]),1): q[order[start:end]]=i
    budget=np.asarray([r['frozen_policy_budget'] for r in z],dtype=np.int16)
    assert np.all(np.isin(budget,[0,50]))
    assert np.array_equal(budget,np.where(q<=3,0,50)),('allocation mismatch',key)
    c0=np.asarray([r['retrieval_correct'] for r in z],dtype=np.int8)
    c20=np.asarray([r['uniform_m20_correct'] for r in z],dtype=np.int8)
    c50=np.asarray([r['uniform_m50_correct'] for r in z],dtype=np.int8)
    cp=np.asarray([r['frozen_policy_correct'] for r in z],dtype=np.int8)
    cm=np.asarray([r['margin_policy_correct'] for r in z],dtype=np.int8)
    assert np.array_equal(cp,np.where(budget==0,c0,c50)),('outcome not reproduced',key)
    return {'q':q,'u':u,'budget':budget,'c0':c0,'c20':c20,'c50':c50,'cp':cp,'cm':cm,
            'idx':np.asarray([r['query_index'] for r in z],dtype=np.int64), 'n':n, 'key':key}

def assess(db,backend,do_group_boot=True):
    groups=[]; rescue=[]; decomp=[]; full=[]
    prepared={}
    for k in sorted(db):
        a=prepare(db[k],k); prepared[k]=a
        retr,ds=k; n=a['n']; q=a['q']; u=a['u']; c0=a['c0'];c20=a['c20'];c50=a['c50'];cp=a['cp'];cm=a['cm']
        assigned=np.where(q<=3,0,50)
        for group in range(1,6):
            s=q==group; N=int(s.sum())
            g={'backend':backend,'retriever':retr,'dataset':ds,'quintile':group,'n':N,
               'entropy_min':float(u[s].min()),'entropy_max':float(u[s].max()),
               'selected_queries_n':int((assigned[s]==50).sum()),
               'allocated_candidate_total':int(assigned[s].sum()),
               'mean_allocated_depth':float(assigned[s].mean()),
               'retrieval_r1_pct':pct(c0[s]),'uniform_m20_r1_pct':pct(c20[s]),
               'uniform_m50_r1_pct':pct(c50[s]),'frozen_policy_r1_pct':pct(cp[s]),
               'policy_minus_m20_pp':pct(cp[s]-c20[s]),
               'm20_minus_retrieval_pp':pct(c20[s]-c0[s]),
               'm50_minus_retrieval_pp':pct(c50[s]-c0[s])}
            if do_group_boot:
                z=np.column_stack((cp[s]-c20[s],c20[s]-c0[s],c50[s]-c0[s]))
                cis=ci(z,seed=SEED + 1000*(['baidu','nordland','st_lucia'].index(ds)+1) + 37*(['boq-dinov2','salad'].index(retr)+1)+group+(10000 if backend=='ALIKED' else 0),B=B_GROUP)
                for c,(low,high) in zip(('policy_minus_m20','m20_minus_retrieval','m50_minus_retrieval'),cis):
                    g[c+'_ci_low_pp']=low;g[c+'_ci_high_pp']=high
            groups.append(g)
            for depth, cmatch in ((20,c20),(50,c50)):
                resc=int(np.sum((c0[s]==0)&(cmatch[s]==1)))
                harm=int(np.sum((c0[s]==1)&(cmatch[s]==0)))
                net=int(np.sum(cmatch[s]-c0[s]))
                assert net==resc-harm,(k,group,depth)
                rescue.append({'backend':backend,'retriever':retr,'dataset':ds,'quintile':group,
                               'depth':depth,'n_denominator':N,'rescue_n':resc,'harm_n':harm,
                               'rescue_rate_pct':100*resc/N,'harm_rate_pct':100*harm/N,
                               'net_n':net,'net_utility_pp':100*net/N})
        skipped=q<=3; selected=~skipped
        X=np.column_stack((np.where(skipped,c0-c20,0),np.where(selected,c50-c20,0),cp-c20))
        exact=X.sum(axis=0)
        assert exact[0]+exact[1]==exact[2]
        bounds=ci(X,seed=SEED+3000*(['baidu','nordland','st_lucia'].index(ds)+1)+99*(['boq-dinov2','salad'].index(retr)+1)+(80000 if backend=='ALIKED' else 0),B=B_CONTRAST)
        dsrow={'backend':backend,'retriever':retr,'dataset':ds,'n_physical_queries':n,
               'skipped_n':int(skipped.sum()),'selected_n':int(selected.sum()),
               'skipped_component_count':int(exact[0]),'selected_component_count':int(exact[1]),
               'policy_minus_m20_count':int(exact[2]),'mean_depth':float(assigned.mean()),
               'skipped_component_pp':float(100*exact[0]/n),
               'selected_component_pp':float(100*exact[1]/n),
               'policy_minus_m20_pp':float(100*exact[2]/n),
               'policy_minus_m50_pp':pct(cp-c50),
               'policy_minus_margin_pp':pct(cp-cm)}
        for label,lim in zip(('skipped_component','selected_component','policy_minus_m20'),bounds):
            dsrow[label+'_ci_low_pp'],dsrow[label+'_ci_high_pp']=lim
        decomp.append(dsrow)
        full.append({'backend':backend,'retriever':retr,'dataset':ds,'n':n,'retrieval_r1_pct':pct(c0),'uniform_m20_r1_pct':pct(c20),'uniform_m50_r1_pct':pct(c50),'policy_r1_pct':pct(cp),'policy_minus_m20_pp':pct(cp-c20),'policy_minus_m50_pp':pct(cp-c50),'mean_depth':float(assigned.mean())})
    return prepared,groups,rescue,decomp,full

def dataset_clustered(prepared,backend):
    """Physical-query clustered per dataset, then fixed-size stratified resampling pooled across datasets.
    The two retriever effects are paired and averaged within each physical query.
    """
    rows_out=[]; vecs=[]
    for ds in ('baidu','nordland','st_lucia'):
        a=prepared[('boq-dinov2',ds)]; b=prepared[('salad',ds)]
        assert np.array_equal(a['idx'],b['idx'])
        z=[]
        for obj in (a,b):
            q=obj['q']; skip=q<=3; select=~skip
            z.append(np.column_stack((np.where(skip,obj['c0']-obj['c20'],0),
                                      np.where(select,obj['c50']-obj['c20'],0),
                                      obj['cp']-obj['c20'],obj['cp']-obj['c50'])))
        v=np.mean(np.stack(z,axis=0),axis=0)
        assert np.allclose(v[:,0]+v[:,1],v[:,2])
        vecs.append(v)
        ends=ci(v,seed=SEED+7000*(['baidu','nordland','st_lucia'].index(ds)+1),B=B_CONTRAST)
        row={'backend':backend,'dataset':ds,'n_physical_queries':len(v),'retrievers':2}
        for i,lab in enumerate(('skipped_component','selected_component','policy_minus_m20','policy_minus_m50')):
            row[lab+'_pp']=pct(v[:,i]);row[lab+'_ci_low_pp'],row[lab+'_ci_high_pp']=ends[i]
        rows_out.append(row)
    means=np.sum([v.sum(axis=0) for v in vecs],axis=0)/sum(len(v) for v in vecs)*100
    # Stratified query-bootstrap: resample each dataset separately; keep original sizes.
    rng=np.random.default_rng(SEED+99001)
    draws=np.zeros((B_CONTRAST,4),dtype=np.float64)
    for v in vecs:
        patterns,counts=np.unique(v,axis=0,return_counts=True)
        draws += rng.multinomial(len(v),counts/len(v),size=B_CONTRAST)@patterns
    draws=draws/sum(len(v) for v in vecs)*100
    qlo,qhi=np.quantile(draws,[.025,.975],axis=0)
    row={'backend':backend,'dataset':'pooled_stratified','n_physical_queries':sum(len(v) for v in vecs),'retrievers':2}
    for i,lab in enumerate(('skipped_component','selected_component','policy_minus_m20','policy_minus_m50')):
        row[lab+'_pp']=float(means[i]);row[lab+'_ci_low_pp']=float(qlo[i]);row[lab+'_ci_high_pp']=float(qhi[i])
    rows_out.append(row)
    return rows_out

def main():
    p=INP/'external_primary_per_query_runner_audit_deduplicated.csv'
    ap=keyed(rows(p))
    expected={('boq-dinov2','baidu'):3208,('boq-dinov2','nordland'):27592,('boq-dinov2','st_lucia'):1464,('salad','baidu'):3208,('salad','nordland'):27592,('salad','st_lucia'):1464}
    assert {k:len(v) for k,v in ap.items()}==expected
    pr,pg,ph,pd,pf=assess(ap,'SuperPoint')
    ak={}
    for ds in ('baidu','nordland','st_lucia'):
        path=INP/f'confirmatory_per_query_aliked_boq_dinov2_{ds}.csv'
        al=rows(path);k=('boq-dinov2',ds);pri=ap[k]
        assert len(al)==len(pri)
        for v,w in zip(al,pri):
            for f in ('retriever','dataset','query_position','query_index','frozen_policy_budget','margin_policy_budget','retrieval_correct'):
                assert v[f]==w[f],(k,f)
            assert abs(v['entropy_uncertainty']-w['entropy_uncertainty'])<1e-12
            assert abs(v['top12_margin']-w['top12_margin'])<1e-12
        ak[k]=al
    ar,ag,ah,ad,af=assess(ak,'ALIKED')
    write(OUT/'baidu_utility_by_group.csv',[r for r in pg+ag if r['dataset']=='baidu'])
    write(OUT/'baidu_rescue_harm.csv',[r for r in ph+ah if r['dataset']=='baidu'])
    write(OUT/'external_contrasts_by_group.csv',[r for r in pg+ag if r['dataset']!='baidu'])
    write(OUT/'external_contrasts_rescue_harm.csv',[r for r in ph+ah if r['dataset']!='baidu'])
    write(OUT/'allocation_loss_decomposition.csv',pd+ad)
    write(OUT/'phase_a_full_condition_checks.csv',pf+af)
    allcluster=dataset_clustered(pr,'SuperPoint')
    write(OUT/'physical_query_clustered_effects.csv',allcluster)
    for grp in ('salad','boq-dinov2'):
        x=[r for r in pd if r['retriever']==grp]
        assert len(x)==3
    actual=next(r for r in allcluster if r['dataset']=='pooled_stratified')
    assert abs(actual['policy_minus_m20_pp'] - (-0.3052938259360278))<1e-10
    # Verify secondary against supplied fixed record using paired physical query aggregation.
    groups_count=[len(a['idx']) for a in pr.values()]
    assert sum(groups_count)==64528
    # Figure: Baidu primary, explicit rescue/harm and net at both depths for each retriever.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11.1,7.0),sharex=True,sharey=True,constrained_layout=True)
    for ri,retr in enumerate(('salad','boq-dinov2')):
        for di,depth in enumerate((20,50)):
            ax=axes[ri,di]
            arr=[r for r in ph if r['dataset']=='baidu' and r['retriever']==retr and r['depth']==depth]
            arr.sort(key=lambda r:r['quintile'])
            xx=np.arange(1,6)
            ax.bar(xx-.24,[z['rescue_rate_pct'] for z in arr],width=.24,label='Rescue',color='#3f86b9')
            ax.bar(xx,[ -z['harm_rate_pct'] for z in arr],width=.24,label='Harm',color='#dd7a56')
            ax.plot(xx+.24,[z['net_utility_pp'] for z in arr],marker='o',markersize=5,linewidth=1.8,label='Net',color='#35554d')
            ax.axhline(0,color='gray',linewidth=.8)
            ax.axvline(3.5,color='black',linestyle='--',linewidth=.9)
            ax.set_title(f'{"SALAD" if retr=="salad" else "BoQ-DINOv2"} | uniform depth {depth}')
            ax.set_xticks(xx)
            ax.set_xlabel('Entropy quintile (1=lowest, 5=highest)')
            if di==0: ax.set_ylabel('Rate among quintile queries (%)')
            if ri==0 and di==1: ax.legend(loc='upper left',fontsize=9)
    fig.suptitle('Baidu: Corrected versus harmed retrieval decisions by entropy group\nDashed line: frozen 0/50 allocation boundary',fontsize=13)
    fig.savefig(OUT/'figures/baidu_rescue_harm.svg',bbox_inches='tight')
    fig.savefig(OUT/'figures/baidu_rescue_harm.pdf',bbox_inches='tight')
    plt.close(fig)
    input_hashes={x.name:sha(x) for x in sorted(INP.iterdir()) if x.is_file()}
    checks={'primary_n':64528,'primary_physical_n':32264,
      'by_retriever_dataset':{f'{r}/{ds}':n for (r,ds),n in expected.items()},
      'all_primary_entropy_quintile_assignments_match_frozen_policy':True,
      'all_primary_frozen_outcomes_match_depth_0_or_50':True,
      'aliked_exact_query_order_count_allocations_retrieval_correct_match':True,
      'aliked_max_entropy_absdiff_less_than':1e-12,
      'paired_physical_query_index_matches_between_retrievers':True,
      'source_sha256':input_hashes,'bootstrap_seed':SEED,'bootstrap_B_group':B_GROUP,
      'bootstrap_B_contrast':B_CONTRAST,
      'bootstrap_design':'paired physical queries; resample within fixed archived dataset, and within group for conditional intervals; cross-retriever effects averaged within physical query, stratified by dataset',
      'point_estimate_policy_minus_m20_pp':actual['policy_minus_m20_pp'],
      'point_estimate_ci_pp':[actual['policy_minus_m20_ci_low_pp'],actual['policy_minus_m20_ci_high_pp']]}
    (OUT/'logs/phase_a_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'n':64528,'clustered':actual,'Baidu_decomposition':[z for z in pd if z['dataset']=='baidu'], 'baidu_rescue_harm':[z for z in ph if z['dataset']=='baidu']},indent=2))

if __name__=='__main__': main()

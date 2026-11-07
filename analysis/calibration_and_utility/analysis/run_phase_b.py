#!/usr/bin/env python3
"""Development-only fixed-setting sensitivity at archived T=0.10. No temperature sweep."""
import csv, json, pathlib, hashlib
from collections import defaultdict
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
SEED=20261106
B=5000
FAMILY={'svox':'svox_family','svox_night':'svox_family','svox_rain':'svox_family',
        'amstertime':'amstertime','sped':'sped',
        'san_francisco_partial_embedded_gps':'san_francisco'}
SETTINGS=[('fraction_20pct_depth50',.2,50),('fraction_30pct_depth50',.3,50),
          ('frozen_fraction_40pct_depth50',.4,50),('fraction_50pct_depth50',.5,50),
          ('fraction_60pct_depth50',.6,50),('fraction_40pct_depth20',.4,20),
          ('fraction_40pct_depth100',.4,100),('uniform_m20',1.0,20),
          ('fraction_20pct_depth100',.2,100)]


def loadcsv(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def csvout(p,rr):
    assert rr
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def ci1d(strata_vals,seed):
    """Query-clustered bootstrap, stratified by original datasets. Arrays of 3-vector paired differences."""
    rng=np.random.default_rng(seed)
    sim=np.zeros((B,3),dtype=float)
    total=sum(len(v) for v in strata_vals)
    for v in strata_vals:
        u,c=np.unique(v,axis=0,return_counts=True)
        sim+=(rng.multinomial(len(v),c/len(v),size=B)@u)
    sim=sim/total*100
    b=np.quantile(sim,[.025,.975],axis=0)
    return b

def main():
    core=loadcsv(ROOT/'inputs/megaloc_per_query_core_for_ci.csv')
    ext=loadcsv(ROOT/'inputs/matching_value_by_uncertainty_quintile.csv')
    ref={(r['dataset'],int(r['uncertainty_quintile']),int(r['budget'])):r for r in ext}
    byds=defaultdict(list)
    for r in core: byds[r['dataset']].append(r)
    assert len(core)==4600 and len({(r['dataset'],r['i']) for r in core})==4600
    assert set(byds)==set(FAMILY)
    arrays={}; qmismatch={}
    for ds,rs in sorted(byds.items()):
        rs.sort(key=lambda r:int(r['i']))
        n=len(rs)
        y={d:np.asarray([int(r['c'+str(d)]) for r in rs],dtype=np.int8) for d in (0,20,50,100)}
        u=np.asarray([float(r['entropy']) for r in rs],dtype=float)
        qarch=np.asarray([int(r['q']) for r in rs],dtype=int)
        assert np.all((qarch>=1)&(qarch<=5))
        for q in range(1,6):
            mask=qarch==q
            for d in (0,20,50,100):
                r=ref[(ds,q,d)]
                assert int(r['n'])==int(mask.sum())
                assert abs(y[d][mask].mean()-float(r['budget_r1']))<1e-12
        order=np.argsort(u,kind='stable')
        qrank=np.zeros(n,dtype=int)
        cuts=[round(k*n/5) for k in (1,2,3,4)]
        for i,(lo,hi) in enumerate(zip([0]+cuts,cuts+[n]),1):qrank[order[lo:hi]]=i
        # Difference between published equal-count archived quintile q and round-boundary groups.
        qmismatch[ds]=int(np.sum(qrank!=qarch))
        arrays[ds]={'n':n,'u':u,'y':y,'order':order,'qarch':qarch,
                    'idx':np.asarray([int(r['i']) for r in rs]),'family':FAMILY[ds]}
    allper=[]; rows=[]; figrows=[]; outputs={}
    for idx,(name,f,depth) in enumerate(SETTINGS):
        cases={}
        # Assign only within dataset; preserve per-dataset integer rounding.
        for ds,a in arrays.items():
            n=a['n']; skip=round((1-f)*n)
            bud=np.zeros(n,dtype=np.int32); bud[a['order'][skip:]]=depth
            assert int(np.sum(bud==depth))==n-skip
            yy=np.where(bud==0,a['y'][0],a['y'][depth]).astype(np.int8)
            cases[ds]={'bud':bud,'yy':yy, 'u20':a['y'][20], 'c0':a['y'][0]}
        if name=='frozen_fraction_40pct_depth50': frozen={ds:cases[ds]['yy'].copy() for ds in cases}
        outputs[name]=cases
    # Compute after all settings, for paired reference differences.
    for si,(name,f,depth) in enumerate(SETTINGS):
        cases=outputs[name]
        for ds,a in sorted(arrays.items()):
            n=a['n']; c=cases[ds];bud=c['bud']; yy=c['yy']
            for k in range(n):
                allper.append({'setting':name,'dataset':ds,'environment_family':a['family'],
                 'i':int(a['idx'][k]),'archived_entropy_T010':float(a['u'][k]),
                 'archived_quintile':int(a['qarch'][k]),'chosen_depth':int(bud[k]),
                 'correct_at_chosen_depth':int(yy[k]),
                 'uniform_m20_correct':int(a['y'][20][k]),
                 'zero_depth_correct':int(a['y'][0][k]),
                 'reference_f040_depth50_correct':int(frozen[ds][k])})
        region={'overall_micro':list(arrays)}
        for ds in arrays:region['dataset:'+ds]=[ds]
        for fam in sorted(set(FAMILY.values())):
            region['family:'+fam]=[ds for ds in arrays if arrays[ds]['family']==fam]
        for region_name,dsets in region.items():
            n=sum(arrays[d]['n'] for d in dsets)
            area_depth=sum(cases[d]['bud'].sum() for d in dsets)
            vals=[np.column_stack((cases[d]['yy']-arrays[d]['y'][20],
                                   cases[d]['yy']-arrays[d]['y'][0],
                                   cases[d]['yy']-frozen[d])).astype(np.float64)
                  for d in dsets]
            bounds=ci1d(vals,seed=SEED+si*97+sum(ord(c) for c in region_name))
            r1=sum(cases[d]['yy'].sum() for d in dsets)/n*100
            u20=sum(arrays[d]['y'][20].sum() for d in dsets)/n*100
            c0=sum(arrays[d]['y'][0].sum() for d in dsets)/n*100
            refscore=sum(frozen[d].sum() for d in dsets)/n*100
            row={'setting':name,'selection_fraction_requested':f,'paid_depth':depth,
                'aggregation':region_name,'n_physical_queries':n,'selected_queries':int(sum((cases[d]['bud']>0).sum() for d in dsets)),
                'realized_mean_candidates':area_depth/n,'recall_at_1_pct':r1,
                'uniform_m20_r1_pct':u20,'retrieval_m0_r1_pct':c0,
                'reference_f040_depth50_r1_pct':refscore}
            for j,lab in enumerate(('minus_uniform_m20','minus_retrieval_m0','minus_reference_f040_depth50')):
                row[lab+'_pp']=float((r1-[u20,c0,refscore][j]))
                row[lab+'_ci_low_pp']=float(bounds[0,j]);row[lab+'_ci_high_pp']=float(bounds[1,j])
            rows.append(row)
            if region_name=='overall_micro':figrows.append(row)
        # Four-family equal-weight descriptive summary; CI stratified per source dataset.
        fs=[r for r in rows if r['setting']==name and r['aggregation'].startswith('family:')]
        assert len(fs)==4
        total=np.array([r['minus_uniform_m20_pp'] for r in fs]).mean()
        famacc=np.array([r['recall_at_1_pct'] for r in fs]).mean()
        famcost=np.array([r['realized_mean_candidates'] for r in fs]).mean()
        rng=np.random.default_rng(SEED+4100+si)
        draws=np.zeros(B)
        for fam in sorted(set(FAMILY.values())):
            dsets=[ds for ds in arrays if arrays[ds]['family']==fam]
            n=sum(arrays[ds]['n'] for ds in dsets)
            by=np.zeros(B)
            for ds in dsets:
                vv=(cases[ds]['yy']-arrays[ds]['y'][20]).astype(float)
                u,c=np.unique(vv,return_counts=True)
                by+=rng.multinomial(len(vv),c/len(vv),size=B)@u
            draws+=by/n/4*100
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append({'setting':name,'selection_fraction_requested':f,'paid_depth':depth,
          'aggregation':'equal_4_family_macro','n_physical_queries':4600,
          'selected_queries':'not_applicable','realized_mean_candidates':famcost,
          'recall_at_1_pct':famacc,'uniform_m20_r1_pct':np.mean([r['uniform_m20_r1_pct'] for r in fs]),
          'retrieval_m0_r1_pct':np.mean([r['retrieval_m0_r1_pct'] for r in fs]),
          'reference_f040_depth50_r1_pct':np.mean([r['reference_f040_depth50_r1_pct'] for r in fs]),
          'minus_uniform_m20_pp':total,'minus_uniform_m20_ci_low_pp':lo,'minus_uniform_m20_ci_high_pp':hi,
          'minus_retrieval_m0_pp':np.mean([r['minus_retrieval_m0_pp'] for r in fs]),
          'minus_retrieval_m0_ci_low_pp':'','minus_retrieval_m0_ci_high_pp':'',
          'minus_reference_f040_depth50_pp':np.mean([r['minus_reference_f040_depth50_pp'] for r in fs]),
          'minus_reference_f040_depth50_ci_low_pp':'','minus_reference_f040_depth50_ci_high_pp':''})
    csvout(ROOT/'phase_b_development_sensitivity.csv',rows)
    csvout(ROOT/'phase_b_per_query_allocation.csv',allper)
    # Figure: result points with uncertainty (paired 95% CI) but not method selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,5.9),constrained_layout=True)
    c={'fraction':'#206c9b','depth':'#b96d45','uniform':'#444444','reference':'#417b4e'}
    for r in figrows:
        name=r['setting']; cost=float(r['realized_mean_candidates']); y=float(r['recall_at_1_pct'])
        cat='reference' if name=='frozen_fraction_40pct_depth50' else ('uniform' if name=='uniform_m20' else ('depth' if 'depth100' in name or 'depth20' in name else 'fraction'))
        err_lo=float(r['minus_uniform_m20_pp'])-float(r['minus_uniform_m20_ci_low_pp'])
        err_hi=float(r['minus_uniform_m20_ci_high_pp'])-float(r['minus_uniform_m20_pp'])
        ax.errorbar(cost,y,yerr=np.array([[err_lo],[err_hi]]),marker='o',linestyle='none',color=c[cat],capsize=3,markersize=6)
        lab=name.replace('fraction_','f=').replace('pct_depth','%,m=').replace('frozen_fraction_40pct_depth50','f=40%,m=50 (ref)')
        ax.annotate(lab,(cost,y),xytext=(6,3),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Realized candidate evaluations per query (mean)')
    ax.set_ylabel('R@1 on original MegaLoc development queries (%)')
    ax.set_title('Archived T=0.10 development sensitivity (exploratory)\nVertical error bars: dataset-stratified paired 95% bootstrap intervals')
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'figures/phase_b_development_accuracy_cost.svg',bbox_inches='tight')
    fig.savefig(ROOT/'figures/phase_b_development_accuracy_cost.pdf',bbox_inches='tight')
    plt.close(fig)
    check={'core_n':len(core),'unique_dataset_i':4600,'datasets':{k:v['n'] for k,v in arrays.items()},
        'archived_quintile_reproduction_mismatch_vs_round_boundaries':qmismatch,
        'original_group_depth_means_reproduced':120,
        'source_hashes':{p.name:sha(p) for p in (ROOT/'inputs/megaloc_per_query_core_for_ci.csv',ROOT/'inputs/matching_value_by_uncertainty_quintile.csv')},
        'seed':SEED,'B':B,'outcomes_available_at_depths':[0,20,50,100],
        'temperature_sweep_status':'BLOCKED - top-100 retrieval distances, and their query/gallery alignment unavailable',
        'interpretation_scope':'original development set, exploratory diagnostic, not independent confirmation'}
    (ROOT/'logs/phase_b_integrity.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({'micro':[r for r in rows if r['aggregation']=='overall_micro'],
                      'macro':[r for r in rows if r['aggregation']=='equal_4_family_macro'],
                      'quintile_differences':qmismatch},indent=2))

if __name__=='__main__':main()

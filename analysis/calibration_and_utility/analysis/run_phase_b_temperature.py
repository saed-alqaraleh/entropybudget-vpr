#!/usr/bin/env python3
"""Genuine T-only development sensitivity using verified top-100 retrieval distance arrays.
Computes entropy from raw distances, never from scalar T=0.10 entropy. No matcher inference.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from collections import defaultdict
from run_phase_b import FAMILY,B,SEED,ci1d,csvout,loadcsv

ROOT=Path(__file__).resolve().parents[1]
TEMPERATURES=[0.05,0.10,0.20,0.50,1.00]
ENTROPY_ATOL = 1e-12

def entropy(d,T):
    d=np.asarray(d,dtype=np.float64)
    mn=d.min(axis=1,keepdims=True);mx=d.max(axis=1,keepdims=True);span=mx-mn
    dn=np.divide(d-mn,span,out=np.zeros_like(d),where=span>1e-12)
    dn=np.where(span<=1e-12,0.5,dn)
    z=-dn/T;z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    return -(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)/np.log(d.shape[1])

def rank_ascending(u):
    order=np.argsort(u,kind='stable')
    out=np.zeros(len(u),dtype=np.int32);out[order]=np.arange(len(u))
    return out

def main():
    core=loadcsv(ROOT/'inputs/megaloc_per_query_core_for_ci.csv')
    byds=defaultdict(list)
    for r in core:byds[r['dataset']].append(r)
    assert len(core)==4600
    states={};matcherr={};hashes={};intra={}
    for ds,rec in sorted(byds.items()):
        rec.sort(key=lambda r:int(r['i']))
        npz=ROOT/f'inputs/retrieval_run02/{ds}.npz'
        hashes[ds]=hashlib.sha256(npz.read_bytes()).hexdigest()
        ar=np.load(npz,allow_pickle=False)
        n=len(rec)
        d=ar['descriptor_distances']
        assert d.shape==(n,100) and ar['candidates'].shape==(n,100)
        assert [int(r['i']) for r in rec]==list(range(n))
        assert len(ar['query_indices'])==n
        en010=entropy(d,.1)
        old=np.asarray([float(r['entropy']) for r in rec])
        maxerr=float(abs(en010-old).max());matcherr[ds]=maxerr
        # NumPy's CPU kernels can round entropy differently at machine precision.
        assert maxerr <= ENTROPY_ATOL, (ds, maxerr)
        order010 = np.argsort(en010, kind='stable')
        assert np.array_equal(order010, np.argsort(old, kind='stable')), (ds, 'T=0.10 rank mismatch')
        vals={0:np.asarray([int(r['c0']) for r in rec]),20:np.asarray([int(r['c20']) for r in rec]),50:np.asarray([int(r['c50']) for r in rec])}
        mask0=np.zeros(n,dtype=bool)
        k=round(.40*n)
        mask0[order010[-k:]]=True
        states[ds]={'d':d,'n':n,'vals':vals,'base':mask0,'entropy010':en010,'rank010':rank_ascending(en010),
                    'query_ids':ar['query_indices'],'k':k,'family':FAMILY[ds]}
    rows=[];per=[];diag=[];figure=[]
    for si,T in enumerate(TEMPERATURES):
        parts={}
        for ds,a in states.items():
            e=entropy(a['d'],T);order=np.argsort(e,kind='stable');k=a['k']
            m=np.zeros(a['n'],dtype=bool);m[order[-k:]]=True
            cp=np.where(m,a['vals'][50],a['vals'][0])
            assert len(cp)==a['n']
            base=np.where(a['base'],a['vals'][50],a['vals'][0])
            overlap=int((m&a['base']).sum())
            rank=rank_ascending(e)
            # Spearman's rho (rank identity in absence of entropy ties)
            rho=float(np.corrcoef(rank,a['rank010'])[0,1])
            d0={'dataset':ds,'temperature':T,'n':a['n'],'selected_n':k,
                'selected_membership_overlap_n':overlap,'selected_membership_overlap_fraction':overlap/k,
                'selected_membership_changed_n':int(np.sum(m!=a['base'])),
                'spearman_entropy_rank_T010':rho,
                'R1_pct':float(cp.mean()*100),
                'difference_vs_T010_pp':float(np.mean(cp-base)*100)}
            diag.append(d0)
            parts[ds]={'cp':cp,'base':base,'c0':a['vals'][0],'c20':a['vals'][20],
                       'm':m,'n':a['n'],'k':k,'overlap':overlap,'rank':rank,'rho':rho}
            for i in range(a['n']):
                per.append({'dataset':ds,'i':i,'original_query_index':int(a['query_ids'][i]),
                            'temperature':T,'entropy':float(e[i]),'entropy_rank':int(rank[i]),
                            'selected_for_depth50':int(m[i]),'budget':int(50*m[i]),
                            'correctness_at_selected_depth':int(cp[i]),
                            'reference_T010_correctness':int(base[i]),
                            'uniform_m20_correctness':int(a['vals'][20][i])})
        regions={'overall_micro':list(states)}
        for ds in states:regions['dataset:'+ds]=[ds]
        for fam in sorted(set(FAMILY.values())):regions['family:'+fam]=[ds for ds in states if FAMILY[ds]==fam]
        for rg,dsets in regions.items():
            n=sum(parts[d]['n'] for d in dsets)
            k=sum(parts[d]['k'] for d in dsets)
            over=sum(parts[d]['overlap'] for d in dsets)
            yy=sum(parts[d]['cp'].sum() for d in dsets)/n*100
            rref=sum(parts[d]['base'].sum() for d in dsets)/n*100
            c20=sum(parts[d]['c20'].sum() for d in dsets)/n*100
            z=[np.column_stack((parts[d]['cp']-parts[d]['base'],
                    parts[d]['cp']-parts[d]['c20'],parts[d]['cp']-parts[d]['c0'])) for d in dsets]
            bounds=ci1d(z,seed=SEED+400000+si*311+sum(ord(v) for v in rg))
            row={'temperature':T,'aggregation':rg,'n_physical_queries':n,'selected_n':k,
                 'realized_mean_candidates':50*k/n,'R1_pct':yy,
                 'R1_at_T010_pct':rref,'uniform_m20_R1_pct':c20,
                 'top40_membership_overlap_fraction':over/k,
                 'top40_membership_changed_n':sum(parts[d]['n']-2*parts[d]['overlap']+2*(parts[d]['overlap']-parts[d]['k']) if False else int(np.sum(parts[d]['m'] != states[d]['base'])) for d in dsets),
                 'mean_spearman_rho_unweighted_datasets':float(np.mean([parts[d]['rho'] for d in dsets]))}
            for j,lab in enumerate(('minus_T010','minus_uniform_m20','minus_retrieval_m0')):
                row[lab+'_pp']=float([yy-rref,yy-c20,yy-sum(parts[d]['c0'].sum() for d in dsets)/n*100][j])
                row[lab+'_ci_low_pp']=float(bounds[0,j]);row[lab+'_ci_high_pp']=float(bounds[1,j])
            rows.append(row)
            if rg=='overall_micro':figure.append(row)
        fs=[r for r in rows if r['temperature']==T and r['aggregation'].startswith('family:')]
        # Equal-family point estimates. Dataset-stratified family bootstrap for paired reference effect.
        rng=np.random.default_rng(SEED+880000+si)
        sim=np.zeros(B)
        for fam in sorted(set(FAMILY.values())):
            dsets=[d for d in states if FAMILY[d]==fam]
            n=sum(parts[d]['n'] for d in dsets)
            sam=np.zeros(B)
            for ds in dsets:
                zz=(parts[ds]['cp']-parts[ds]['base']).astype(float)
                u,c=np.unique(zz,return_counts=True)
                sam+=rng.multinomial(len(zz),c/len(zz),size=B)@u
            sim+=(sam/n/4*100)
        lo,hi=np.quantile(sim,[.025,.975])
        rows.append({'temperature':T,'aggregation':'equal_4_family_macro','n_physical_queries':4600,'selected_n':'not_applicable',
             'realized_mean_candidates':float(np.mean([r['realized_mean_candidates'] for r in fs])),
             'R1_pct':float(np.mean([r['R1_pct'] for r in fs])),
             'R1_at_T010_pct':float(np.mean([r['R1_at_T010_pct'] for r in fs])),
             'uniform_m20_R1_pct':float(np.mean([r['uniform_m20_R1_pct'] for r in fs])),
             'top40_membership_overlap_fraction':float(np.mean([r['top40_membership_overlap_fraction'] for r in fs])),
             'top40_membership_changed_n':'not_applicable','mean_spearman_rho_unweighted_datasets':float(np.mean([r['mean_spearman_rho_unweighted_datasets'] for r in fs])),
             'minus_T010_pp':float(np.mean([r['minus_T010_pp'] for r in fs])),
             'minus_T010_ci_low_pp':float(lo),'minus_T010_ci_high_pp':float(hi),
             'minus_uniform_m20_pp':float(np.mean([r['minus_uniform_m20_pp'] for r in fs])),
             'minus_uniform_m20_ci_low_pp':'','minus_uniform_m20_ci_high_pp':'',
             'minus_retrieval_m0_pp':float(np.mean([r['minus_retrieval_m0_pp'] for r in fs])),
             'minus_retrieval_m0_ci_low_pp':'','minus_retrieval_m0_ci_high_pp':''})
    csvout(ROOT/'phase_b_temperature_sensitivity.csv',rows)
    csvout(ROOT/'phase_b_temperature_per_query_allocation.csv',per)
    csvout(ROOT/'phase_b_temperature_ordering_diagnostics.csv',diag)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8.3,5.0),constrained_layout=True)
    xx=[r['temperature'] for r in figure]; yy=[r['R1_pct'] for r in figure]
    err=[[r['minus_T010_pp']-r['minus_T010_ci_low_pp'] for r in figure],
         [r['minus_T010_ci_high_pp']-r['minus_T010_pp'] for r in figure]]
    ax.errorbar(xx,yy,yerr=err,marker='o',capsize=4,linewidth=1.6,color='#2f6586')
    ax.set_xscale('log');ax.set_xticks(TEMPERATURES);ax.set_xticklabels([str(a) for a in TEMPERATURES]);
    ax.set_xlabel('Retrieval entropy temperature (allocation only)')
    ax.set_ylabel('R@1 (%) on 4,600 MegaLoc development queries')
    ax.set_title('Top 40% at depth 50; genuine temperature sweep\nPaired 95% CIs versus T=0.10, held at the original 20-candidate mean')
    ax.axhline(figure[1]['R1_pct'],linestyle='--',color='#888888',linewidth=.8)
    ax.grid(alpha=.23)
    fig.savefig(ROOT/'figures/phase_b_temperature_accuracy.svg',bbox_inches='tight')
    fig.savefig(ROOT/'figures/phase_b_temperature_accuracy.pdf',bbox_inches='tight')
    plt.close(fig)
    check={'archive_alignment':'original archived i equals retrieval NPZ row offset',
       'max_abs_entropy_error_T010':matcherr,
       'entropy_abs_tolerance':ENTROPY_ATOL,
       'T010_ordering_matches_archive':True,
       'original_top100_distances_sha256':hashes,'n_original_queries':4600,
       'temperatures':TEMPERATURES,'temperature_affects':'retrieval uncertainty and allocation only',
       'scorer_is_frozen':'reused c0/c50 outcomes, no model or matching rerun',
       'bootstrap_seed_base':SEED,'bootstrap_replicates':B,
       'status':'COMPLETED - T=0.10 entropy within tolerance and exact archived ordering'}
    (ROOT/'logs/phase_b_temperature_integrity.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({'micro':figure,'diagnostics':diag},indent=2))

if __name__=='__main__':main()

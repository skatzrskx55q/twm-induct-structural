"""Descriptive clustered statistics; open dev diagnostics only, not sealed test."""
import json,collections,math
from pathlib import Path
import numpy as np,torch,sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from structural_induction import StructuralNode
from semantic_n2_experiment import read_n2,Buckets
SEEDS=(17,29,43); SPLITS=('dev_L2','dev_L3');MODES=('conditioned','static','ablate','random','E36')

def loadrows(split,mode,seed):
    p=ROOT/'results'/f'rollout_{mode}_seed{seed}_traces.jsonl'
    return {v['id']:v for v in (json.loads(s) for s in p.read_text().splitlines()) if v['split']==split}

def paired_diff(split,mode_a,mode_b,h=16,reps=10000):
    bylaw=collections.defaultdict(list)
    for seed in SEEDS:
        a=loadrows(split,mode_a,seed);b=loadrows(split,mode_b,seed)
        assert a.keys()==b.keys()
        for ep,aa in a.items():
            bb=b[ep]
            ia=int(all(aa['trace_edge_node'][h-1]));ib=int(all(bb['trace_edge_node'][h-1]))
            bylaw[aa['law']].append(ia-ib)
    assert len(bylaw)==20 and all(len(z)==12 for z in bylaw.values())
    a=np.array([np.mean(x) for x in bylaw.values()]);rng=np.random.default_rng(20261104)
    boots=a[rng.integers(len(a),size=(reps,len(a)))].mean(-1)*100
    return {'difference_percentage_points':float(a.mean()*100),'cluster_95_ci_pp':[float(v) for v in np.quantile(boots,[.025,.975])],
            'n_laws':len(a),'n_paired_world_seed':sum(len(x) for x in bylaw.values()), 'favorable_laws':int((a>0).sum()),
            'unfavorable_laws':int((a<0).sum()),'tied_laws':int((a==0).sum()),
            'percentile_bootstrap':'descriptive exploratory, repeated open dev'}

@torch.no_grad()
def weight_specialization(split):
    data=Buckets(read_n2()[split]);out={}
    for seed in SEEDS:
        m=StructuralNode(conditioned=True).eval()
        ck=torch.load(ROOT/'checkpoints'/f'Structural_conditioned_seed{seed}.pt',weights_only=True,map_location='cpu')
        m.load_state_dict(ck['model'])
        arr=[];laws=[];weights=[]
        for n,b in data.packed.items():
            for i in range(0,len(b['ns']),20):
                _,w,err=m.make_context(b['ns'][i:i+20],b['ny'][i:i+20]);weights.append(w.numpy())
                laws.extend([x[0] for x in b['meta'][i:i+20]])
        weights=np.concatenate(weights)
        bylaw=collections.defaultdict(list)
        for k,w in zip(laws,weights):bylaw[k].append(w)
        # L1 distinguishability of weights across same law vs different law.
        same=[];diff=[]
        keys=list(bylaw)
        for law,ww in bylaw.items():
            for i in range(len(ww)):
                for j in range(i+1,len(ww)):same.append(float(abs(ww[i]-ww[j]).sum()))
        for j in range(len(keys)-1):
            diff.append(float(abs(np.mean(bylaw[keys[j]],axis=0)-np.mean(bylaw[keys[j+1]],axis=0)).sum()))
        entropy=-(weights*np.log(weights.clip(1e-12))).sum(-1)
        out[str(seed)]={'mean_entropy':float(np.mean(entropy)),'effective_trees':float(np.mean(np.exp(entropy))),
                         'mean_max_weight':float(weights.max(axis=1).mean()),'same_law_L1':float(np.mean(same)),
                         'between_law_mean_L1':float(np.mean(diff)), 'n_episodes':len(weights)}
    return out

def main():
    summary={}
    for split in SPLITS:
        totals={mode:{str(h):collections.Counter() for h in (1,4,8,16)} for mode in MODES}
        for mode in MODES:
            for seed in SEEDS:
                record=json.loads((ROOT/'results'/f'rollout_{mode}_seed{seed}.json').read_text())
                for h,agg in record['horizons'][split].items():totals[mode][h].update(agg)
        summary[split]={'counts':{mode:{k:dict(v) for k,v in hs.items()} for mode,hs in totals.items()},
                        'comparisons_h16':{f'conditioned_minus_{o}':paired_diff(split,'conditioned',o) for o in ('static','ablate','E36')},
                        'mechanism_weight_diagnostics':weight_specialization(split)}
    (ROOT/'results'/'STATISTICAL_AND_MECHANISM_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for split,v in summary.items():
        print(split,{k:{h:vv['full'] for h,vv in m.items()} for k,m in v['counts'].items()})
        print('COMPARE',split,json.dumps(v['comparisons_h16'],ensure_ascii=False))
        print('MECHANISM',split,json.dumps(v['mechanism_weight_diagnostics'],ensure_ascii=False))
if __name__=='__main__':main()

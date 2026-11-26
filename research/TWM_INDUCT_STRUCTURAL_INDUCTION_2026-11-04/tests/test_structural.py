"""No oracle in runtime: all tests use observed history/query. Labels scorer-only."""
import sys,os,json
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from structural_induction import StructuralNode
from structural_rollout import make_predictor
from joint_event_pilot import permute_row
from semantic_n2_experiment import read_n2,Buckets
from structural_experiment import cache_queries,load_edge
from e36_meta_ridge import Model as E36Model

torch.set_num_threads(1)

def sample():
    r=read_n2()['dev_L2']
    return next(iter(next(iter(r.values()))))

def load(seed=17):
    edge=load_edge(seed)
    ck=torch.load(ROOT/'checkpoints'/f'Structural_conditioned_seed{seed}.pt',weights_only=True,map_location='cpu')
    node=StructuralNode(conditioned=True).eval();node.load_state_dict(ck['model'])
    return node,edge

def test_partition_normalizes():
    m=StructuralNode();x=torch.randn(3,5,30)
    h=m.membership(x)
    assert h.shape==(3,5,24,4)
    assert torch.allclose(h.sum(-1),torch.ones(3,5,24),atol=1e-6)

def test_no_future_labels_or_law_id_leakage():
    node,edge=load();r=sample()
    n=r['n_nodes'];predict=make_predictor(node,edge,r['history'],n)
    p=predict(r['query_nodes'],r['query_edges'],r['future_actions'][0])
    altered=dict(r,future_nodes=[int(1-x) for x in r['future_nodes']],future_edges=[int(1-x) for x in r['future_edges']],law_id='ADVERSARIAL')
    a=predict(altered['query_nodes'],altered['query_edges'],altered['future_actions'][0])
    assert all(np.array_equal(q,v) for q,v in zip(p,a))

def test_support_labels_are_used():
    m,_=load();s=torch.rand(2,48,30);q=torch.rand(2,6,30)
    support=torch.zeros(2,48)
    assert torch.max(abs(m(s,q,support)-m(s,q,1-support))).item()>0.01

def test_history_selection_is_active():
    m,_=load();r=sample();n=r['n_nodes'];data=Buckets({n:[r]})
    d=data.packed[n]
    _,w,_=m.make_context(d['ns'],d['ny'])
    assert float(w.std().item())>0.001
    # Disable the selection WITHOUT changing trained parameters.
    with torch.no_grad():
        m.conditioned=False
        _,u,_=m.make_context(d['ns'],d['ny'])
    assert torch.max(abs(w-u)).item()>0.01

def test_permutation_equivariance():
    m,e=load();r=sample();n=r['n_nodes'];perm=np.random.default_rng(2026).permutation(n)
    rp=permute_row(r,perm)
    p0=make_predictor(m,e,r['history'],n)(r['query_nodes'],r['query_edges'],r['future_actions'][0])
    p1=make_predictor(m,e,rp['history'],n)(rp['query_nodes'],rp['query_edges'],rp['future_actions'][0])
    assert np.array_equal(p1[0],p0[0][perm]);assert np.array_equal(p1[1].reshape(n,n),p0[1].reshape(n,n)[np.ix_(perm,perm)])

def test_structural_inference_on_new_sizes():
    node,edge=load()
    for n in (11,12):
        rng=np.random.default_rng(n); hist=[];x=rng.integers(0,2,size=n,dtype=np.int8)
        e=rng.integers(0,2,size=(n,n),dtype=np.int8);np.fill_diagonal(e,0)
        for _ in range(8):
            act=int(rng.integers(n));xn=x.copy();en=e.copy()
            # Synthetic observed transitions only, not claiming valid causal accuracy.
            xx=x.copy();xx[int(rng.integers(n))]^=1
            ee=e.copy();i,j=rng.choice(n,size=2,replace=False);ee[i,j]^=1
            hist.append(dict(action=act,nodes_before=xn.tolist(),nodes_after=xx.tolist(),edges_before=en.reshape(-1).tolist(),edges_after=ee.reshape(-1).tolist()))
            x,e=xx,ee
        pr=make_predictor(node,edge,hist,n)
        a,b=pr(x,e.reshape(-1),0)
        assert len(a)==n and len(b)==n*n and np.all(np.diag(b.reshape(n,n))==0)

def test_rollout_H1_matches_one_step():
    for seed in (17,29,43):
        for mode in ('conditioned','static'):
            h1=json.loads((ROOT/'results'/f'structural_{mode}_seed{seed}_H1.json').read_text())
            roll=json.loads((ROOT/'results'/f'rollout_{mode}_seed{seed}.json').read_text())
            for split in ('dev_L2','dev_L3'):
                one=h1[split]['score'];a=roll['horizons'][split]['1']
                assert (one['full'],one['edge'],one['node'])==(a['full'],a['edge'],a['node'])

def test_original_E36_open_N2_metrics_reproduced():
    sums={sp:{h:0 for h in (1,16)} for sp in ('dev_L2','dev_L3')}
    for seed in (17,29,43):
        r=json.loads((ROOT/'results'/f'rollout_E36_seed{seed}.json').read_text())
        for sp in sums:
            for h in sums[sp]:sums[sp][h]+=r['horizons'][sp][str(h)]['full']
    assert sums=={'dev_L2':{1:181,16:84},'dev_L3':{1:175,16:108}}

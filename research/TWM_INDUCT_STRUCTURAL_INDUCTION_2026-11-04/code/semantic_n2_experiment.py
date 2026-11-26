#!/usr/bin/env python3
"""Fair open-N2 semantic L2 diagnostic. NO law oracle in model or inference.
Original N2 data already public; train/dev only; labels isolated in evaluator/trainer.
N2 historical names are preserved without redefinition. n-bucket batching supports 6..9.
"""
import argparse, collections, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from joint_event_pilot import encode_rows,JointEventModel,build_node_query,seed_all
from e36_meta_ridge import Model as E36Model
from e34_induction import infer as CART_infer

TRAIN='train';DEVS=('dev_L2','dev_L3','ID_state')

def read_n2():
    inputs=[json.loads(z) for z in (ROOT/'reference/N2_MODEL_INPUTS.jsonl').read_text().splitlines()]
    labels={z['episode_id']:z for z in (json.loads(x) for x in (ROOT/'reference/N2_PRIVATE_LABELS.jsonl').read_text().splitlines())}
    meta={z['episode_id']:z for z in (json.loads(x) for x in (ROOT/'reference/N2_PRIVATE_EPISODE_META.jsonl').read_text().splitlines())}
    assert len(inputs)==len(labels)==len(meta)==len({r['episode_id'] for r in inputs})
    d=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in inputs:
        if r['split'] not in (TRAIN,*DEVS):continue
        y=labels[r['episode_id']];m=meta[r['episode_id']];assert r['split']==m['split']
        assert len(r['history'])==len(r['future_actions'])==16
        assert len(y['future_edges'])==len(y['future_nodes'])==16
        # N2 already provides immutable 16-step observed history; last eight for E36 parity.
        data={k:r[k] for k in ('n_nodes','history','query_nodes','query_edges','future_actions')}
        data['history']=data['history'][-8:]
        data['future_actions']=[data['future_actions'][0]]
        data['future_edges']=y['future_edges'][0]
        data['future_nodes']=y['future_nodes'][0]
        data['episode_id']=r['episode_id']
        data['law_id']=m['law_key'] # tracked by scorer, never passed into neural forward
        d[r['split']][r['n_nodes']].append(data)
    return d

def audit_semantics(groups):
    reg=json.loads((ROOT/'reference/N2_PRIVATE_LAW_REGISTRY.json').read_text())
    a={sp:set(reg['splits'][sp]) for sp in (TRAIN,*DEVS)}
    tr_edge=set(reg['mechanisms']['edge']['train']);tr_node=set(reg['mechanisms']['node']['train'])
    from lawbench_v0 import mechanism_key
    def coverage(name):
        edges={mechanism_key(reg['laws'][k]['edge']) for k in a[name]}
        nodes={mechanism_key(reg['laws'][k]['node']) for k in a[name]}
        return {'laws':len(a[name]),'edge_mechanisms':len(edges),'node_mechanisms':len(nodes),'new_edge_mechanisms':sorted(edges-tr_edge),'new_node_mechanisms':sorted(nodes-tr_node)}
    result={name:coverage(name) for name in (TRAIN,*DEVS)}
    assert len(a[TRAIN]&a['dev_L2'])==0
    assert not result['dev_L2']['new_edge_mechanisms'] and not result['dev_L2']['new_node_mechanisms']
    assert result['dev_L3']['new_edge_mechanisms'] or result['dev_L3']['new_node_mechanisms']
    assert a['ID_state'] <= a[TRAIN]
    result['counts']={s:sum(len(q) for q in groups[s].values()) for s in (TRAIN,*DEVS)}
    result['intersection_full_laws_train_dev_L2']=len(a[TRAIN]&a['dev_L2'])
    result['method']='Frozen N2 registry and observational input/meta join; semantic signature uniqueness inherited from original N2 audit, not a new exhaustive semantic proof.'
    return result

class Buckets:
    def __init__(self,rows):
        self.rows=rows
        self.packed={n:encode_rows(v) for n,v in rows.items()}
        self.sizes={n:len(v) for n,v in rows.items()}
        self.nums=sorted(rows)
        self.probs=np.array([self.sizes[k] for k in self.nums],dtype=np.float64)
        self.probs/=self.probs.sum()
    def batch(self,rng,B):
        n=int(rng.choice(self.nums,p=self.probs))
        ids=torch.from_numpy(rng.integers(self.sizes[n],size=B,dtype=np.int64))
        return {k:v[ids] for k,v in self.packed[n].items() if k!='meta'}

@torch.no_grad()
def e36_predict(model,b):
    pe=model.edge(b['es'],b['eq'],b['ey'])
    nq,_,_=build_node_query(b['qn'],b['qb'],pe,b['qa'])
    pn=model.node(b['ns'],nq,b['ny'])
    return pe,pn

def calc_metrics(pe,pn,ye,yn):
    pe=pe>=.5;pn=pn>=.5;ye=ye.bool();yn=yn.bool()
    eo=(pe==ye).all(-1);no=(pn==yn).all(-1);fo=eo&no
    neg=(~ye).sum().item()+(~yn).sum().item()
    preserving=((~pe)&(~ye)).sum().item()+((~pn)&(~yn)).sum().item()
    return dict(n=len(eo),full=int(fo.sum()),edge=int(eo.sum()),node=int(no.sum()),edge_only_errors=int((~eo&no).sum()),node_only_errors=int((eo&~no).sum()),both_errors=int((~eo&~no).sum()),preservation_bits=int(preserving),preservation_total=int(neg))

@torch.no_grad()
def evaluate(model,d,kind):
    model.eval()
    agg=collections.Counter();bylaw=collections.defaultdict(collections.Counter)
    for n,packed in d.packed.items():
        for i in range(0,d.sizes[n],16):
            b={k:v[i:i+16] for k,v in packed.items() if k!='meta'}
            pe,pn=e36_predict(model,b) if kind=='E36' else model(b)
            stats=calc_metrics(pe,pn,b['et'],b['nt']);agg.update(stats)
            exact=((pe>=.5)==b['et'].bool()).all(-1)&((pn>=.5)==b['nt'].bool()).all(-1)
            for m,hit in zip(packed['meta'][i:i+16],exact):
                law=m[0];bylaw[law]['n']+=1;bylaw[law]['full']+=int(hit.item())
    return {'counts':dict(agg),'per_law':dict(bylaw)}

def train_E36(model,data,seed,edge_steps,node_steps,batch=16):
    rng_edge=np.random.default_rng(seed);rng_node=np.random.default_rng(seed+10000)
    optE=torch.optim.AdamW(model.edge.parameters(),lr=.0012,weight_decay=.001)
    optN=torch.optim.AdamW(model.node.parameters(),lr=.0012,weight_decay=.001)
    logs=[];start=time.monotonic()
    for k in range(edge_steps):
        b=data.batch(rng_edge,batch)
        pr=model.edge(b['es'],b['eq'],b['ey'])
        loss=F.binary_cross_entropy(pr.clamp(1e-6,1-1e-6),b['et'])
        optE.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.edge.parameters(),3.);optE.step()
        if k in (0,edge_steps//2,edge_steps-1):logs.append({'head':'edge','step':k+1,'bce':round(float(loss.item()),6)})
    model.edge.eval()
    # Recompute node query features from model's OWN edge predictions, consistent with historical E36.
    nodecache={}
    for n,p in data.packed.items():
        with torch.no_grad():
            probs=[]
            for i in range(0,data.sizes[n],16):probs.append(model.edge(p['es'][i:i+16],p['eq'][i:i+16],p['ey'][i:i+16]))
            a=torch.cat(probs)
            q,_,_=build_node_query(p['qn'],p['qb'],a,p['qa']);nodecache[n]=q.detach()
    for k in range(node_steps):
        n=int(rng_node.choice(data.nums,p=data.probs));ids=torch.from_numpy(rng_node.integers(data.sizes[n],size=batch,dtype=np.int64))
        b=data.packed[n];pr=model.node(b['ns'][ids],nodecache[n][ids],b['ny'][ids])
        loss=F.binary_cross_entropy(pr.clamp(1e-6,1-1e-6),b['nt'][ids]);optN.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.node.parameters(),3.);optN.step()
        if k in (0,node_steps//2,node_steps-1):logs.append({'head':'node','step':k+1,'bce':round(float(loss.item()),6)})
    return {'log':logs,'elapsed_s':round(time.monotonic()-start,3)}

def train_Joint(model,data,seed,steps,batch=16):
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.001)
    logs=[];start=time.monotonic()
    model.train()
    for k in range(steps):
        b=data.batch(rng,batch)
        pe,pn=model(b)
        le=F.binary_cross_entropy(pe.clamp(1e-6,1-1e-6),b['et'])
        ln=F.binary_cross_entropy(pn.clamp(1e-6,1-1e-6),b['nt'])
        loss=le+ln;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step()
        if k in (0,steps//2,steps-1) or (k+1)%100==0:
            logs.append({'step':k+1,'edge_bce':round(float(le.item()),6),'node_bce':round(float(ln.item()),6)});print('JOINT_PROGRESS',seed,logs[-1],round(time.monotonic()-start,2),flush=True)
    return {'log':logs,'elapsed_s':round(time.monotonic()-start,3)}

def eval_cart(groups):
    out={};start=time.monotonic()
    for split in DEVS:
        stat=collections.Counter()
        for n,rows in groups[split].items():
            for r in rows:
                em,nm=CART_infer(r['history'],r['query_nodes'],r['query_edges'],r['future_actions'][0])
                eb=np.asarray(r['query_edges'],dtype=np.int8).reshape(n,n);xb=np.asarray(r['query_nodes'],dtype=np.int8)
                e=eb^em;x=xb^nm
                ed=np.array_equal(e.ravel(),r['future_edges']);nd=np.array_equal(x,r['future_nodes'])
                stat.update(n=1,full=int(ed and nd),edge=int(ed),node=int(nd))
        out[split]=dict(stat)
    return {'metrics':out,'elapsed_seconds':round(time.monotonic()-start,2),'nature':'ALGORITHMIC CART, NOT NEURAL'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='17,29,43')
    ap.add_argument('--edge-steps',type=int,default=750)
    ap.add_argument('--node-steps',type=int,default=750)
    ap.add_argument('--joint-steps',type=int,default=1500)
    ap.add_argument('--train-models',choices=['both','E36','Joint','none'],default='both')
    ap.add_argument('--skip-cart',action='store_true')
    ap.add_argument('--out',default=str(ROOT/'results'))
    args=ap.parse_args();torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('READ HISTORICAL N2',flush=True)
    g=read_n2();reg=audit_semantics(g)
    (out/'SEMANTIC_SPLIT_AUDIT.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2))
    data={s:Buckets(g[s]) for s in (TRAIN,*DEVS)}
    print('SPLIT CHECK',reg['counts'],flush=True)
    if not args.skip_cart:
        cart=eval_cart(g)
        (out/'CART_DEV_OPEN.json').write_text(json.dumps(cart,ensure_ascii=False,indent=2))
        print('CART',cart,flush=True)
    if args.train_models=='none':return
    for seed in map(int,args.seeds.split(',')):
        for kind in (('E36','Joint') if args.train_models=='both' else (args.train_models,)):
            seed_all(seed)
            model=E36Model() if kind=='E36' else JointEventModel()
            t0=time.monotonic()
            info=train_E36(model,data[TRAIN],seed,args.edge_steps,args.node_steps) if kind=='E36' else train_Joint(model,data[TRAIN],seed,args.joint_steps)
            metrics={s:evaluate(model,data[s],kind) for s in DEVS}
            res={'model':kind,'seed':seed,'steps':({'edge':args.edge_steps,'node':args.node_steps} if kind=='E36' else {'joint':args.joint_steps}),
                 'train_info':info,'metrics':metrics,'params':sum(p.numel() for p in model.parameters()),
                 'elapsed_total_s':round(time.monotonic()-t0,2),'status':'OPEN_DEVELOPMENT_NO_PROMOTION'}
            (out/f'{kind}_seed{seed}_dev_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
            torch.save({'model':model.state_dict(),'seed':seed,'name':kind,'training':info,'status':'OPEN_DEV_NOT_PROMOTED'},out/f'{kind}_seed{seed}_N2_train.pt')
            print('MODEL_RESULT',kind,seed,'timing',res['elapsed_total_s'],{k:v['counts'] for k,v in metrics.items()},flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Autonomous rollout on historically OPEN N2 dev_L2/dev_L3 using learned N2 E36 weights.
Support fitted once, predictions use own previous state, future truth scorer-only.
"""
import json,collections,sys,time,argparse
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from e36_meta_ridge import Model
from joint_event_pilot import JointEventModel
from joint_cached_predictor_reference import AutonomousPredictor
from e34_induction import local_features,node_features,infer as cart_infer

HORIZONS=(1,4,8,16)

@torch.no_grad()
def make_predictor(model,history,n):
    non_diag=~np.eye(n,dtype=bool)
    ex=[];ey=[];nx=[];ny=[]
    for tr in history[-8:]:
        e,_=local_features(tr['nodes_before'],tr['edges_before'],tr['action'])
        ex.append(e[non_diag]);ey.append((np.asarray(tr['edges_before'],dtype='i1').reshape(n,n)^np.asarray(tr['edges_after'],dtype='i1').reshape(n,n))[non_diag])
        nx.append(node_features(tr['nodes_before'],tr['edges_before'],tr['edges_after'],tr['action']))
        ny.append(np.asarray(tr['nodes_before'],dtype='i1')^np.asarray(tr['nodes_after'],dtype='i1'))
    xs=[torch.from_numpy(np.concatenate(ex))[None],torch.from_numpy(np.concatenate(nx))[None]]
    ys=[torch.as_tensor(np.concatenate(ey),dtype=torch.float32)[None],torch.as_tensor(np.concatenate(ny),dtype=torch.float32)[None]]
    weights=[]
    for head,x,y in zip([model.edge,model.node],xs,ys):
        X=head._embed(x);xt=X.transpose(1,2);d=X.shape[-1]
        la=torch.nn.functional.softplus(head.log_lambda)+0.02
        weights.append(torch.linalg.solve(xt@X+la*torch.eye(d)[None],xt@y[...,None]))
    def pred(nodes,edges,actor):
        with torch.no_grad():
            e,_=local_features(nodes,edges,actor)
            def predict_one(head,features,w):
                Q=torch.from_numpy(np.asarray(features,dtype='f4'))[None]
                raw=(head._embed(Q)@w).squeeze(-1)
                return (raw>=.5).squeeze(0).numpy().astype('i1')
            em=predict_one(model.edge,e[non_diag],weights[0])
            ep=np.asarray(edges,dtype='i1').reshape(n,n).copy();ep[non_diag]^=em
            u=node_features(nodes,edges,ep,actor)
            nm=predict_one(model.node,u,weights[1])
        return np.asarray(nodes,dtype='i1')^nm,ep.reshape(-1)
    return pred

def run(seed,kind='E36'):
    torch.set_num_threads(4)
    inputs=[json.loads(x) for x in (ROOT/'reference/N2_MODEL_INPUTS.jsonl').read_text().splitlines()]
    labels={y['episode_id']:y for y in (json.loads(z) for z in (ROOT/'reference/N2_PRIVATE_LABELS.jsonl').read_text().splitlines())}
    meta={y['episode_id']:y for y in (json.loads(z) for z in (ROOT/'reference/N2_PRIVATE_EPISODE_META.jsonl').read_text().splitlines())}
    model=None
    if kind=='Joint':
        ck=torch.load(ROOT/'results'/f'Joint_seed{seed}_N2_train.pt',map_location='cpu',weights_only=True)
        model=JointEventModel().eval();model.load_state_dict(ck['model'])
    if kind=='E36':
        ck=torch.load(ROOT/'results'/f'E36_seed{seed}_N2_train.pt',map_location='cpu',weights_only=True)
        model=Model().eval();model.load_state_dict(ck['model'])
    stats={s:{str(h):collections.Counter() for h in HORIZONS} for s in ['dev_L2','dev_L3']}
    per_law=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(collections.Counter)))
    records=[];t0=time.monotonic()
    for inp in inputs:
        s=inp['split']
        if s not in stats:continue
        label=labels[inp['episode_id']]; m=meta[inp['episode_id']]
        n=inp['n_nodes'];x=np.asarray(inp['query_nodes'],dtype=np.int8);e=np.asarray(inp['query_edges'],dtype=np.int8)
        predict=make_predictor(model,inp['history'],n) if kind=='E36' else (AutonomousPredictor(model,inp['history']).step if kind=='Joint' else None)
        pathok=True; trace=[]
        for step,(act,true_n,true_e) in enumerate(zip(inp['future_actions'],label['future_nodes'],label['future_edges']),1):
            if kind in ('E36','Joint'):x,e=predict(x,e,act)
            else:
                em,nm=cart_infer(inp['history'][-8:],x,e,act)
                ee=np.asarray(e,dtype=np.int8).reshape(n,n).copy();ee^=em
                x=x^nm;e=ee.reshape(-1)
            good_n=bool(np.array_equal(x,true_n));good_e=bool(np.array_equal(e,true_e))
            pathok &= good_n and good_e
            if step in HORIZONS:
                q=stats[s][str(step)];q.update(n=1,exact=int(good_e and good_n),node_exact=int(good_n),edge_exact=int(good_e),path_exact=int(pathok))
                v=per_law[s][str(step)][m['law_key']]
                v['n']+=1;v['exact']+=int(good_e and good_n)
            trace.append([int(good_n),int(good_e)])
        records.append({'id':inp['episode_id'],'split':s,'seed':seed,'model':kind,'trace_node_edge_exact':trace})
    out={'status':'OPEN_N2_DEV_NOT_SEALED','model':kind,'seed':seed,'elapsed_s':round(time.monotonic()-t0,2),'score':{s:{h:dict(q) for h,q in stats[s].items()} for s in stats},'per_law':{s:{h:{k:dict(v) for k,v in q.items()} for h,q in by.items()} for s,by in per_law.items()}}
    (ROOT/'results'/f'rollout_{kind}_seed{seed}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    with (ROOT/'results'/f'rollout_{kind}_seed{seed}_traces.jsonl').open('w') as f:
        for v in records:f.write(json.dumps(v,separators=(',',':'))+'\n')
    print('ROLLOUT',kind,seed,out['elapsed_s'],out['score'],flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='17,29,43');p.add_argument('--model',choices=['E36','CART','Joint'],default='E36');a=p.parse_args()
    for s in map(int,a.seeds.split(',')):run(s,kind=a.model)

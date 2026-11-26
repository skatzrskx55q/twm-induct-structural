"""Open N2 autonomous H1/H4/H8/H16, including support-conditioned ablation.
At inference no law ID, future correct states, or simulator is read.
The evaluator reads future labels separately ONLY to score predictions.
"""
import argparse,collections,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from e36_meta_ridge import Model as E36Model
from e34_induction import local_features,node_features
from structural_induction import StructuralNode

@torch.no_grad()
def make_predictor(node, edge, history,n, mode='conditioned',ridge_node=None):
    offdiag=~np.eye(n,dtype=bool)
    es=[];ey=[];ns=[];ny=[]
    for tr in history[-8:]:
        es0,_=local_features(tr['nodes_before'],tr['edges_before'],tr['action'])
        eb=np.asarray(tr['edges_before'],dtype=np.int8).reshape(n,n)
        ea=np.asarray(tr['edges_after'],dtype=np.int8).reshape(n,n)
        es.append(es0[offdiag]);ey.append((eb^ea)[offdiag])
        ns.append(node_features(tr['nodes_before'],eb,ea,tr['action']))
        ny.append(np.asarray(tr['nodes_before'],dtype=np.int8)^np.asarray(tr['nodes_after'],dtype=np.int8))
    xe=torch.from_numpy(np.concatenate(es).astype('f4'))[None]
    ye=torch.from_numpy(np.concatenate(ey).astype('f4'))[None]
    X=edge._embed(xe)
    lam=torch.nn.functional.softplus(edge.log_lambda)+.02
    ew=torch.linalg.solve(X.transpose(1,2)@X+lam*torch.eye(X.shape[-1])[None],X.transpose(1,2)@ye[...,None])
    sn=torch.from_numpy(np.concatenate(ns).astype('f4'))[None]
    yn=torch.from_numpy(np.concatenate(ny).astype('f4'))[None]
    if ridge_node is None:
        context=node.make_context(sn,yn)
    else:
        XX=ridge_node._embed(sn)
        la=torch.nn.functional.softplus(ridge_node.log_lambda)+.02
        context=torch.linalg.solve(XX.transpose(1,2)@XX+la*torch.eye(XX.shape[-1])[None],XX.transpose(1,2)@yn[...,None])
    def step(nodes,edges,action):
        edge_f,_=local_features(nodes,edges,action)
        q=torch.from_numpy(edge_f[offdiag].astype('f4'))[None]
        pr=((edge._embed(q)@ew).squeeze(-1)>=.5).squeeze(0).numpy().astype('i1')
        nextedges=np.asarray(edges,dtype='i1').reshape(n,n).copy();nextedges[offdiag]^=pr
        nf=node_features(nodes,edges,nextedges,action)
        qt=torch.from_numpy(nf.astype('f4'))[None]
        if ridge_node is None:
            p=node.predict_context(qt,context)
        else:
            raw=(ridge_node._embed(qt)@context).squeeze(-1)
            p=torch.sigmoid((raw-.5)*ridge_node.temperature.clamp(.5,12.))
        flips=(p>=.5).squeeze(0).numpy().astype('i1')
        return (np.asarray(nodes,dtype='i1')^flips),nextedges.reshape(-1)
    return step

def read_open_data(splits):
    inputs=[json.loads(s) for s in (ROOT/'reference/N2_MODEL_INPUTS.jsonl').read_text().splitlines()]
    targets={x['episode_id']:x for x in (json.loads(s) for s in (ROOT/'reference/N2_PRIVATE_LABELS.jsonl').read_text().splitlines())}
    meta={x['episode_id']:x for x in (json.loads(s) for s in (ROOT/'reference/N2_PRIVATE_EPISODE_META.jsonl').read_text().splitlines())}
    return [(r,targets[r['episode_id']],meta[r['episode_id']]) for r in inputs if r['split'] in splits]

def run(seed,mode,episodes,horizons=(1,4,8,16)):
    torch.set_num_threads(1)
    ck=torch.load(ROOT/'checkpoints'/f'E36_seed{seed}_N2_train.pt',map_location='cpu',weights_only=True)
    e36=E36Model().eval();e36.load_state_dict(ck['model'])
    node=None;edge=e36.edge;rn=None
    if mode!='E36':
        if mode=='random':
            torch.manual_seed(seed)
            node=StructuralNode(conditioned=True).eval()
        else:
            ck2=torch.load(ROOT/'checkpoints'/f'Structural_{"conditioned" if mode=="ablate" else mode}_seed{seed}.pt',map_location='cpu',weights_only=True)
            node=StructuralNode(conditioned=(mode=='conditioned')).eval();node.load_state_dict(ck2['model'])
            if mode=='ablate':node.conditioned=False
    else:
        node=e36.node;rn=e36.node
    stats={s:{str(h):collections.Counter() for h in horizons} for s in ['dev_L2','dev_L3']}
    rows=[];t0=time.monotonic()
    for inp,lab,metadata in episodes:
        split=inp['split'];n=inp['n_nodes']
        predict=make_predictor(node,edge,inp['history'],n,ridge_node=rn)
        x=np.asarray(inp['query_nodes'],dtype='i1');e=np.asarray(inp['query_edges'],dtype='i1');trajectory=[];path=True
        for i,(a,t_node,t_edge) in enumerate(zip(inp['future_actions'],lab['future_nodes'],lab['future_edges']),1):
            x,e=predict(x,e,a)
            edgeok=bool(np.array_equal(e,t_edge));nodeok=bool(np.array_equal(x,t_node));full=edgeok and nodeok;path=path and full
            trajectory.append([int(edgeok),int(nodeok)])
            if i in horizons:stats[split][str(i)].update(n=1,full=int(full),edge=int(edgeok),node=int(nodeok),path=int(path))
        rows.append({'id':inp['episode_id'],'law':metadata['law_key'],'split':split,'seed':seed,'mode':mode,'n_nodes':n,'trace_edge_node':trajectory})
    report={'seed':seed,'mode':mode,'horizons':{s:{h:dict(v) for h,v in dat.items()} for s,dat in stats.items()},
        'episodes':len(episodes),'elapsed_s':round(time.monotonic()-t0,3),'status':'HISTORICAL_OPEN_N2_ONLY'}
    p=ROOT/'results'/f'rollout_{mode}_seed{seed}'
    (p.with_suffix('.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (p.parent/(p.name+'_traces.jsonl')).open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('ROLLOUT',mode,seed,report['horizons'],'seconds',report['elapsed_s'],flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',default='conditioned,static,ablate,E36');p.add_argument('--seeds',default='17,29,43');a=p.parse_args()
    eps=read_open_data(['dev_L2','dev_L3'])
    for mode in a.modes.split(','):
        for seed in [int(q) for q in a.seeds.split(',')]:run(seed,mode,eps)

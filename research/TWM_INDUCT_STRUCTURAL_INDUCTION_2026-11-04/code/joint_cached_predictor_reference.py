#!/usr/bin/env python3
"""Rollout on HISTORICAL OPEN N2 (test_L3 and dev_L3); NOT an independent sealed exam.
Exactly eight observed transitions, frozen fitted context, autonomous query updates.
Never reads labels in the model predictor or uses intermediate oracle states.
"""
from pathlib import Path
import sys,json,time,argparse,collections
import torch,numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'source'))
from joint_event_pilot import JointEventModel,encode_rows,local_features,node_features_torch,build_node_query
from torch.nn import functional as F

def fit_ridge(head,X,y):
    dim=X.shape[-1];xt=X.transpose(1,2)
    la=F.softplus(head.log_lambda)+.02
    eye=torch.eye(dim,device=X.device,dtype=X.dtype)[None]
    return torch.linalg.solve(xt@X+la*eye,xt@y[...,None])

def predict_ridge(head,Q,W):
    raw=(Q@W).squeeze(-1)
    return torch.sigmoid((raw-.5)*head.temperature.clamp(.5,12))

class AutonomousPredictor:
    """Model sees ONLY observed history (8) and evolving self-generated state/action."""
    def __init__(self,model,history):
        self.model=model
        n=len(history[0]['nodes_before']);self.n=n
        # Placeholder query initializes shape; no 'future_nodes' or 'future_edges' passed.
        r={'n_nodes':n,'history':history[-8:],'query_nodes':history[-1]['nodes_after'],
           'query_edges':history[-1]['edges_after'],'future_actions':[history[-1]['action']]}
        b=encode_rows([r]);self.p=n*(n-1)
        self.non_diag=~np.eye(n,dtype=bool)
        with torch.no_grad():
            se=model.event(b['es']);XE=model.edge.embed(b['es'],se)
            self.We=fit_ridge(model.edge,XE,b['ey'])
            sctx=model.node_context(se.reshape(1,8,self.p,-1),b['ey'].reshape(1,8,self.p),n).reshape(1,8*n,-1)
            XN=model.node.embed(b['ns'],sctx)
            self.Wn=fit_ridge(model.node,XN,b['ny'])
    def predict_proba(self,nodes,edges,action):
        n=self.n
        e,_=local_features(nodes,edges,action)
        eq=torch.from_numpy(np.asarray(e[self.non_diag],dtype='f4'))[None]
        qnodes=torch.from_numpy(np.asarray(nodes,dtype='f4'))[None]
        qedges=torch.from_numpy(np.asarray(edges,dtype='f4').reshape(1,n,n))
        qa=torch.tensor([action],dtype=torch.long)
        with torch.no_grad():
            eemb=self.model.event(eq)
            edgeq=self.model.edge.embed(eq,eemb)
            pe=predict_ridge(self.model.edge,edgeq,self.We)
            nodeq,_,_=build_node_query(qnodes,qedges,pe,qa)
            nqctx=self.model.node_context(eemb[:,None],pe[:,None],n)[:,0]
            pnode=predict_ridge(self.model.node,self.model.node.embed(nodeq,nqctx),self.Wn)
            return pe[0].numpy(),pnode[0].numpy()
    def step(self,nodes,edges,actor):
        pe,pn=self.predict_proba(nodes,edges,actor)
        n=self.n;e=np.asarray(edges,dtype='i1').reshape(n,n).copy()
        e[self.non_diag]^=(pe>=.5).astype('i1')
        return np.asarray(nodes,dtype='i1')^(pn>=.5).astype('i1'),e.reshape(-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--ablation',action='store_true');ap.add_argument('--splits',default='dev_L3,test_L3');ap.add_argument('--max-episodes',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(4)
    files=ROOT/'data'; xlist=[json.loads(s) for s in (files/'N2_MODEL_INPUTS.jsonl').open()]
    ymap={x['episode_id']:x for x in (json.loads(s) for s in (files/'N2_PRIVATE_LABELS.jsonl').open())}
    source=(ROOT/'results'/'ablation'/f'seed{args.seed}' if args.ablation else ROOT/'results'/f'full_seed{args.seed}')/f'joint_event_seed{args.seed}_pilot.pt'
    ck=torch.load(source,weights_only=True,map_location='cpu')
    model=JointEventModel(detach_node_events=args.ablation).eval();model.load_state_dict(ck['model'])
    splits=args.splits.split(',');stats={s:{str(h):{'n':0,'exact':0,'node_exact':0,'edge_exact':0,'path_exact':0} for h in (1,4,8,16)} for s in splits}
    testcount=0;max_err=0.;raw=[]
    st=time.monotonic()
    for x in xlist:
        if x['split'] not in splits: continue
        if args.max_episodes and testcount>=args.max_episodes:break
        label=ymap[x['episode_id']]
        assert x['episode_id']==label['episode_id']
        p=AutonomousPredictor(model,x['history']);nodes=np.asarray(x['query_nodes'],dtype='i1');edges=np.asarray(x['query_edges'],dtype='i1')
        # First-query predictor cache parity with original model forward
        if testcount<5:
            query_stub={'n_nodes':x['n_nodes'],'history':x['history'][-8:],'query_nodes':nodes.tolist(),'query_edges':edges.tolist(),'future_actions':[x['future_actions'][0]]}
            inp=encode_rows([query_stub]);a,b=model(inp);a2,b2=p.predict_proba(nodes,edges,x['future_actions'][0]);
            max_err=max(max_err,float(np.max(np.abs(a.detach().numpy()[0]-a2))),float(np.max(np.abs(b.detach().numpy()[0]-b2))))
        path=True;trace=[]
        for t,(actor,truthn,truthe) in enumerate(zip(x['future_actions'],label['future_nodes'],label['future_edges']),1):
            nodes,edges=p.step(nodes,edges,actor)
            ngood=np.array_equal(nodes,truthn);egood=np.array_equal(edges,truthe)
            path&=bool(ngood and egood)
            trace.append({'h':t,'pred_nodes':nodes.tolist(),'pred_edges':edges.tolist(),'node_exact':bool(ngood),'edge_exact':bool(egood),'path_exact':bool(path)})
            if t in (1,4,8,16):
                point=stats[x['split']][str(t)];point['n']+=1;point['exact']+=int(ngood and egood);point['node_exact']+=int(ngood);point['edge_exact']+=int(egood);point['path_exact']+=int(path)
        raw.append({'episode_id':x['episode_id'],'split':x['split'],'seed':args.seed,'trace':trace})
        testcount+=1
        if testcount%80==0:print('PROGRESS',args.seed,testcount,round(time.monotonic()-st,1),flush=True)
    assert max_err<1e-5, ('cache forward parity',max_err)
    res={'status':'HISTORICAL_OPEN_NOT_SEALED','ablation':args.ablation,'seed':args.seed,'splits':stats,'n_episodes':testcount,'self_test_cache_max_error':max_err,'elapsed_seconds':round(time.monotonic()-st,1)}
    out=ROOT/'results'/(f'N2_open_ablation_seed{args.seed}.json' if args.ablation else f'N2_open_seed{args.seed}.json');out.write_text(json.dumps(res,ensure_ascii=False,indent=2))
    raw_path=ROOT/'results'/(f'N2_RAW_ablation_seed{args.seed}.jsonl' if args.ablation else f'N2_RAW_seed{args.seed}.jsonl')
    with raw_path.open('w') as ff:
        for item in raw:ff.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
    print('RESULT',args.seed,stats,'CACHE_PARITY',max_err,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""E36: Meta-learned episodic ridge solver with generic pairwise features.
No rule identifiers/formulas/teacher oracle in inference or training features.
Training targets from observed train queries only. Sequential edge->node.
"""
from pathlib import Path
import json, math, os, sys, random, time, argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from e35_neural_context import pack,tensors,evaluate
from e35_sequential import prep_node30

def seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)

class Solver(nn.Module):
    def __init__(self,d,hid=24):
        super().__init__(); self.d=d
        ii,jj=torch.triu_indices(d,d,offset=1)
        self.register_buffer('ii',ii);self.register_buffer('jj',jj)
        dim=d+len(ii)
        self.feat=nn.Sequential(nn.Linear(dim,72),nn.Tanh(),nn.Linear(72,hid),nn.Tanh())
        self.log_lambda=nn.Parameter(torch.tensor(-0.3))
        self.temperature=nn.Parameter(torch.tensor(4.0))
    def _embed(self,x):
        z=torch.cat([x,x[...,self.ii]*x[...,self.jj]],dim=-1)
        h=self.feat(z)
        return torch.cat([torch.ones_like(h[...,:1]),h],dim=-1)
    def forward(self,s,q,sy):
        X=self._embed(s)
        Q=self._embed(q)
        # Fit contextual regression using ONLY observed support labels.
        xt=X.transpose(1,2)
        dim=X.shape[-1]
        lam=F.softplus(self.log_lambda)+0.02
        eye=torch.eye(dim,dtype=X.dtype,device=X.device)[None,:,:]
        mat=xt@X + lam*eye
        rhs=xt@sy[...,None]
        W=torch.linalg.solve(mat,rhs)
        raw=(Q@W).squeeze(-1)
        return torch.sigmoid((raw-0.5)*self.temperature.clamp(0.5,12.0))

class Model(nn.Module):
    def __init__(self,edge=True):
        super().__init__()
        self.edge=Solver(12)
        self.node=Solver(30)
    def forward(self,b):
        es,eq,ey,et,ns,nq,ny,nt=b
        return self.edge(es,eq,ey), self.node(ns,nq,ny)

def train_solver(model,splitpack,which,seed,steps=750,batch_size=16,lr=0.0012):
    arr=tensors(splitpack)
    if which=='edge': select=(0,1,2,3); unit=model.edge
    else: select=(4,5,6,7);unit=model.node
    rng=np.random.default_rng(seed+(0 if which=='edge' else 10000))
    opt=torch.optim.AdamW(unit.parameters(),lr=lr,weight_decay=.001)
    unit.train();logs=[];t0=time.monotonic()
    for k in range(steps):
        ix=torch.from_numpy(rng.integers(len(arr[0]),size=batch_size,dtype=np.int64))
        s,q,sy,yy=[arr[j][ix] for j in select]
        out=unit(s,q,sy)
        loss=F.binary_cross_entropy(out.clamp(1e-6,1-1e-6), yy)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(unit.parameters(),3.0);opt.step()
        if k==0 or (k+1)%250==0:
            row={'step':k+1,'bce':round(float(loss.item()),6),'elapsed_s':round(time.monotonic()-t0,1), 'lambda':round(float(F.softplus(unit.log_lambda).item()),5)}
            logs.append(row); print('TRAIN',seed,which,row,flush=True)
    return opt.state_dict(),logs

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--steps',type=int,default=750);pa.add_argument('--seeds',default='17,29,43'); pa.add_argument('--batch',type=int,default=16);args=pa.parse_args()
    paths={k:ROOT/'E34'/'INPUTS'/(k+'_records.json') for k in ['train','known_val','paired_known']};paths['N39_open']=ROOT/'E34'/'REF'/'TWM_N39'/'DATA'/'N39_HELD_RECORDS.json'
    rows={k:json.loads(p.read_text())['rows'] for k,p in paths.items()}
    orig={k:pack(rs) for k,rs in rows.items()}
    out=ROOT/'E36_RESULTS';out.mkdir(exist_ok=True)
    mets={};raw={}
    for seed in map(int,args.seeds.split(',')):
        seeds(seed);model=Model()
        osedge,el=train_solver(model,orig['train'],'edge',seed,steps=args.steps,batch_size=args.batch)
        packs={k:prep_node30(rows[k],orig[k],model.edge) for k in rows}
        osnode,nl=train_solver(model,packs['train'],'node',seed,steps=args.steps,batch_size=args.batch)
        ck={'seed':seed,'steps_edge':args.steps,'steps_node':args.steps,'model':model.state_dict(),'optimizer_edge':osedge,'optimizer_node':osnode,'log_edge':el,'log_node':nl}
        torch.save(ck,out/f'E36_seed{seed}.pt')
        mets[str(seed)]={'train_log_edge':el,'train_log_node':nl,'splits':{}}
        model.eval()
        for name in ['train','known_val','paired_known','N39_open']:
            res,pr=evaluate(model,packs[name],batch_size=64)
            mets[str(seed)]['splits'][name]=res
            for key,val in pr.items(): raw[f'{seed}__{name}__{key}']=val
            print('EVAL',seed,name,res['full_exact'],'/',res['n'],'edge',res['edge_exact'],'node',res['node_exact'],'pairs',res['exact_pairs'],flush=True)
    (out/'E36_METRICS.json').write_text(json.dumps(mets,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'E36_RAW.npz',**raw)

if __name__=='__main__':main()

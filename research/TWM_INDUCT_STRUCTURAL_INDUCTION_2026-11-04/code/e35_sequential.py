#!/usr/bin/env python3
"""E35 sequential causal graph variant; neural edge predictions determine node query features. No oracle graph used."""
import json,sys,time,os,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from e35_neural_context import Embed,pack,tensors,evaluate
sys.path.insert(0,str(ROOT/'E34'/'CODE'))
from e34_induction import node_features

class SequentialNet(nn.Module):
    def __init__(self):
        super().__init__();self.edge=Embed(12);self.node=Embed(30)
    def forward(self, batch):
        es,eq,ey,et,ns,nq,ny,nt=batch
        return self.edge(es,eq,ey),self.node(ns,nq,ny)

def prep_node30(rows,prev_pack,edge_model):
    edge_model.eval();np.random.seed(0)
    arr=tensors(prev_pack); edges=[]
    with torch.no_grad():
        for st in range(0,len(rows),128):
            pp=edge_model(arr[0][st:st+128],arr[1][st:st+128],arr[2][st:st+128]);edges.extend((pp>=.5).numpy().astype('i1'))
    es,eq,ey,et,_,_,ny,nt=prev_pack['arrays']
    NS=[];NQ=[]
    for row,pred in zip(rows,edges):
        n=row['n_nodes'];valid=~np.eye(n,dtype=bool)
        sup=[]
        for t in row['history']:
            sup.append(node_features(t['nodes_before'],t['edges_before'],t['edges_after'],t['action']))
        eb=np.array(row['query_edges'],dtype='i1').reshape(n,n);ep=eb.copy();ep[valid]^=pred
        qn=node_features(row['query_nodes'],eb,ep,row['future_actions'][0])
        NS.append(np.concatenate(sup));NQ.append(qn)
    return {'arrays':[es,eq,ey,et,np.asarray(NS,'f4'),np.asarray(NQ,'f4'),ny,nt], 'meta':prev_pack['meta']}

def main():
    torch.set_num_threads(4)
    paths={k:ROOT/'E34'/'INPUTS'/(k+'_records.json') for k in ['train','known_val','paired_known']}
    paths['N39_open']=ROOT/'E34'/'REF'/'TWM_N39'/'DATA'/'N39_HELD_RECORDS.json'
    rows={k:json.load(open(p))['rows'] for k,p in paths.items()}
    orig={k:pack(r) for k,r in rows.items()}
    out=ROOT/'RESULTS';met={};raw={}
    for seed in (17,29,43):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        previous=torch.load(out/f'E35_seed{seed}.pt',map_location='cpu',weights_only=False)
        model=SequentialNet();model.edge.load_state_dict({k[5:]:v for k,v in previous['model'].items() if k.startswith('edge.')})
        packseq={k:prep_node30(rows[k],orig[k],model.edge) for k in paths}
        train=tensors(packseq['train']);rng=np.random.default_rng(seed+500);optimizer=torch.optim.AdamW(model.node.parameters(),lr=.0015,weight_decay=.001)
        log=[];st=time.time()
        model.edge.eval();
        for step in range(1200):
            ix=torch.from_numpy(rng.integers(len(train[0]),size=32,dtype=np.int64));batch=[v[ix] for v in train]
            pr=model.node(batch[4],batch[5],batch[6]);loss=F.binary_cross_entropy(pr.clamp(1e-6,1-1e-6),batch[7]);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.node.parameters(),3);optimizer.step()
            if step==0 or (step+1)%300==0:
                q={'step':step+1,'node_bce':round(loss.item(),6),'elapsed':round(time.time()-st,2)};log.append(q);print('TRAIN_SEQ',seed,q,flush=True)
        torch.save({'seed':seed,'model':model.state_dict(),'optimizer_node':optimizer.state_dict(),'log':log,'edge_from':f'E35_seed{seed}.pt'},out/f'E35_sequential_seed{seed}.pt')
        met[str(seed)]={'train_log':log,'splits':{}}
        for split in paths:
            scores,rs=evaluate(model,packseq[split]);met[str(seed)]['splits'][split]=scores
            for k,v in rs.items():raw[f'{seed}__{split}__{k}']=v
            print('SCORE_SEQ',seed,split,scores['full_exact'],'/',scores['n'],'edge',scores['edge_exact'],'node',scores['node_exact'],'pairs',scores['exact_pairs'],flush=True)
    (out/'E35_SEQUENTIAL_METRICS.json').write_text(json.dumps(met,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'E35_SEQUENTIAL_RAW.npz',**raw)
if __name__=='__main__':main()

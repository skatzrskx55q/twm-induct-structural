"""Equal-budget training/evaluation of TWM INDUCT structural conditional operator.
Entrypoint receives train/dev label data in an evaluator; model forward never sees future.
"""
import argparse, collections, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from e36_meta_ridge import Model as E36Model
from joint_event_pilot import encode_rows, build_node_query
from semantic_n2_experiment import read_n2,Buckets
from structural_induction import StructuralNode

def set_seed(seed):
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def load_edge(seed):
    ck=torch.load(ROOT/'checkpoints'/f'E36_seed{seed}_N2_train.pt',map_location='cpu',weights_only=True)
    mod=E36Model().eval()
    mod.load_state_dict(ck['model'])
    for p in mod.edge.parameters():p.requires_grad_(False)
    return mod.edge

@torch.no_grad()
def cache_queries(buckets,edge):
    caches={}
    for n,b in buckets.packed.items():
        chunks=[]
        for i in range(0,buckets.sizes[n],32):
            sl=slice(i,i+32)
            pe=edge(b['es'][sl],b['eq'][sl],b['ey'][sl])
            q,_,_=build_node_query(b['qn'][sl],b['qb'][sl],pe,b['qa'][sl]);chunks.append(q.detach())
        caches[n]=torch.cat(chunks)
    return caches

def train_one(seed,mode,train,steps=850,batchsize=16, lr=.002):
    set_seed(seed+(0 if mode=='conditioned' else 1000))
    # Identical initialization and minibatch sequence across variants of same seed.
    torch.manual_seed(seed)
    node=StructuralNode(conditioned=mode=='conditioned')
    edge=load_edge(seed)
    caches=cache_queries(train,edge)
    opt=torch.optim.AdamW(node.parameters(),lr=lr,weight_decay=0.001)
    rng=np.random.default_rng(seed+5000)
    t0=time.monotonic(); logs=[];node.train()
    for k in range(steps):
        n=int(rng.choice(train.nums,p=train.probs))
        ids=torch.from_numpy(rng.integers(train.sizes[n],size=batchsize,dtype=np.int64))
        dat=train.packed[n]
        pred=node(dat['ns'][ids],caches[n][ids],dat['ny'][ids])
        loss=F.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),dat['nt'][ids]);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(node.parameters(),3.);opt.step()
        if k in (0,steps//2,steps-1):
            logs.append({'step':k+1,'bce':float(loss.item()),'elapsed_s':round(time.monotonic()-t0,2)})
            print('TRAIN',mode,seed,logs[-1],flush=True)
    node.eval()
    ck={'model':node.state_dict(),'seed':seed,'mode':mode,'steps':steps,'batch':batchsize,'lr':lr,
        'edge_sha256':hashlib.sha256((ROOT/'checkpoints'/f'E36_seed{seed}_N2_train.pt').read_bytes()).hexdigest(),
        'logs':logs,'status':'RESEARCH_NOT_PROMOTED'}
    torch.save(ck,ROOT/'checkpoints'/f'Structural_{mode}_seed{seed}.pt')
    return node,edge,logs

@torch.no_grad()
def evaluate_one(node,edge,buckets,split,seed,mode):
    caches=cache_queries(buckets,edge)
    totals=collections.Counter();by_law=collections.defaultdict(lambda:collections.Counter())
    rows=[]
    node.eval()
    for n,dat in buckets.packed.items():
        for i in range(0,buckets.sizes[n],16):
            sl=slice(i,i+16)
            pe=edge(dat['es'][sl],dat['eq'][sl],dat['ey'][sl])
            pn=node(dat['ns'][sl],caches[n][sl],dat['ny'][sl])
            E=((pe>=.5)==dat['et'][sl].bool()).all(-1).cpu().numpy()
            N=((pn>=.5)==dat['nt'][sl].bool()).all(-1).cpu().numpy()
            for j,(e,nd) in enumerate(zip(E,N)):
                full=bool(e and nd);law=dat['meta'][i+j][0]
                by_law[law].update(n=1,full=int(full),edge=int(e),node=int(nd))
                totals.update(n=1,full=int(full),edge=int(e),node=int(nd))
                rows.append({'law':law,'n_nodes':n,'edge':int(e),'node':int(nd),'full':int(full),'seed':seed,'mode':mode,'split':split})
    return {'score':dict(totals),'by_law':{k:dict(v) for k,v in by_law.items()},'rows':rows}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',default='17,29,43');parser.add_argument('--steps',type=int,default=850);parser.add_argument('--modes',default='conditioned,static');args=parser.parse_args()
    set_seed(17)
    groups=read_n2();data={s:Buckets(groups[s]) for s in ['train','dev_L2','dev_L3']}
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    for seed in [int(s) for s in args.seeds.split(',')]:
        for mode in args.modes.split(','):
            node,edge,log=train_one(seed,mode,data['train'],args.steps)
            scores={}
            for split in ('dev_L2','dev_L3'):
                scores[split]=evaluate_one(node,edge,data[split],split,seed,mode)
                print('SCORE',mode,seed,split,scores[split]['score'],flush=True)
            (out/f'structural_{mode}_seed{seed}_H1.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2))
    print('DONE',flush=True)
if __name__=='__main__':main()

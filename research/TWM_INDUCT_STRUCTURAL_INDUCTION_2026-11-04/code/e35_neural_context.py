#!/usr/bin/env python3
"""E35 neural nonparametric episodic rule induction.
At inference receives only a list of eight observed transitions, current state,
current action. No rule ID / rule formula / target future is consumed.
For each edge and node, it learns embeddings and uses label-bearing observed
transitions to form a differentiable episodic attention memory.
"""
import os,sys,json,random,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'E34'/'CODE'))
from e34_induction import local_features

def pack(rows):
    ee=[]; eq=[];ey=[];et=[]; nn0=[];nq=[];ny=[];nt=[];m=[]
    for r in rows:
        n=r['n_nodes']; assert n==4
        diagonal=~np.eye(n,dtype=bool)
        es=[];el=[];ns=[];nl=[]
        for tr in r['history']:
            e,u=local_features(tr['nodes_before'],tr['edges_before'],tr['action'])
            es.append(e[diagonal]);ns.append(u)
            el.append((np.array(tr['edges_before'],dtype='i1').reshape(n,n)^np.array(tr['edges_after'],dtype='i1').reshape(n,n))[diagonal]);
            nl.append(np.array(tr['nodes_before'],dtype='i1')^np.array(tr['nodes_after'],dtype='i1'))
        E,U=local_features(r['query_nodes'],r['query_edges'],r['future_actions'][0]);
        ee.append(np.concatenate(es)); eq.append(E[diagonal]);ey.append(np.concatenate(el));
        et.append((np.array(r['query_edges'],dtype='i1').reshape(n,n)^np.array(r['future_edges'],dtype='i1').reshape(n,n))[diagonal]);
        nn0.append(np.concatenate(ns));nq.append(U);ny.append(np.concatenate(nl));nt.append(np.array(r['query_nodes'],dtype='i1')^np.array(r['future_nodes'],dtype='i1'))
        m.append((r.get('law_id','unknown'),r.get('pair_id',''),r.get('history_hash','')))
    out=[ee,eq,ey,et,nn0,nq,ny,nt]
    return {'arrays':[np.asarray(a,dtype=np.float32) for a in out], 'meta':m}

class Embed(nn.Module):
    def __init__(self,d,embed=24):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,48), nn.Tanh(),nn.Linear(48,embed))
        self.log_temperature=nn.Parameter(torch.tensor(0.0))
        self.log_prior=nn.Parameter(torch.tensor(-2.0))
    def forward(self,support,query,sy):
        # cosine-scaled similarity; learn feature interactions; no label leakage.
        ks=F.normalize(self.net(support),dim=-1)
        qq=F.normalize(self.net(query),dim=-1)
        score=torch.einsum('bqe,bse->bqs',qq,ks) * (self.log_temperature.exp()*5)
        # pooled class probability as supervised attention over known transition outcomes.
        wt=F.softmax(score,dim=-1)
        prob=torch.einsum('bqs,bs->bq',wt,sy)
        # fixed smooth clipping in loss only, exact classification via >=0.5
        return prob

class EpisodeNet(nn.Module):
    def __init__(self):
        super().__init__();self.edge=Embed(12);self.node=Embed(10)
    def forward(self, batch):
        es,eq,ey,et,ns,nq,ny,nt=batch
        return self.edge(es,eq,ey),self.node(ns,nq,ny)

def tensors(packaged):return [torch.from_numpy(a) for a in packaged['arrays']]

def evaluate(model, packed,device='cpu',batch_size=128):
    arr=tensors(packed);pred_e=[];pred_n=[];true_e=[];true_n=[]
    model.eval()
    with torch.no_grad():
        for k in range(0,len(arr[0]),batch_size):
            cur=[v[k:k+batch_size].to(device) for v in arr];a,b=model(cur)
            pred_e.append((a>=.5).cpu().numpy());pred_n.append((b>=.5).cpu().numpy());true_e.append(cur[3].cpu().numpy().astype('bool'));true_n.append(cur[7].cpu().numpy().astype('bool'))
    pe=np.concatenate(pred_e);pn=np.concatenate(pred_n);ye=np.concatenate(true_e);yn=np.concatenate(true_n)
    edgeok=np.all(pe==ye,axis=1);nodeok=np.all(pn==yn,axis=1);fullok=edgeok&nodeok
    bylaw={}
    for ok,(law,*_) in zip(fullok,packed['meta']):
        q=bylaw.setdefault(law,{'n':0,'full':0});q['n']+=1;q['full']+=int(ok)
    pairs=None
    if len(fullok)%2==0 and all(packed['meta'][i][1] and packed['meta'][i][1]==packed['meta'][i+1][1] for i in range(0,len(fullok),2)):
        pairs=int(sum(fullok[i] and fullok[i+1] for i in range(0,len(fullok),2)))
    return {'n':len(fullok),'full_exact':int(fullok.sum()),'edge_exact':int(edgeok.sum()),'node_exact':int(nodeok.sum()),'false_edge_flip':int((pe&~ye).sum()),'missed_edge_flip':int((~pe&ye).sum()),'per_law':bylaw,'exact_pairs':pairs}, {'pe':pe.astype('i1'),'pn':pn.astype('i1'),'ye':ye.astype('i1'),'yn':yn.astype('i1'),'fullok':fullok.astype('i1')}

def train(seed,trainpack,steps=1000,batch_size=32,lr=.0015,dev='cpu'):
    torch.set_num_threads(4); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=EpisodeNet().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.001)
    arr=tensors(trainpack);n=len(arr[0]);rng=np.random.default_rng(seed)
    log=[];t0=time.time()
    for i in range(steps):
        ix=torch.from_numpy(rng.integers(n,size=batch_size,dtype=np.int64))
        batch=[v[ix].to(dev) for v in arr]
        a,b=model(batch)
        # BCE on direct observational future target, no teacher oracle or formula.
        le=F.binary_cross_entropy(a.clamp(1e-6,1-1e-6),batch[3]); ln=F.binary_cross_entropy(b.clamp(1e-6,1-1e-6),batch[7]);loss=le+ln
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if i==0 or (i+1)%200==0:
            v={'step':i+1,'edge_bce':round(le.item(),6),'node_bce':round(ln.item(),6),'elapsed':round(time.time()-t0,1)};log.append(v);print('TRAIN',seed,v,flush=True)
    return model,opt.state_dict(),log

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--seeds',default='17,29,43');ap.add_argument('--batch',type=int,default=32);a=ap.parse_args()
    datadir=ROOT/'E34'/'INPUTS';out=ROOT/'RESULTS';out.mkdir(exist_ok=True)
    splits={}
    for k in ['train','known_val','paired_known']:
        p=datadir/(k+'_records.json');r=json.load(open(p))['rows']; print('PACK',k,len(r),flush=True);splits[k]=pack(r)
    p=ROOT/'E34'/'REF'/'TWM_N39'/'DATA'/'N39_HELD_RECORDS.json';splits['N39_open']=pack(json.load(open(p))['rows'])
    allmetrics={};raw={}
    for seed in map(int,a.seeds.split(',')):
        model,opt,log=train(seed,splits['train'],steps=a.steps,batch_size=a.batch)
        torch.save({'model':model.state_dict(),'optimizer':opt,'seed':seed,'steps':a.steps,'log':log},out/f'E35_seed{seed}.pt')
        allmetrics[str(seed)]={'train_log':log,'splits':{}}
        for name in ['train','known_val','paired_known','N39_open']:
            metrics,rs=evaluate(model,splits[name]);allmetrics[str(seed)]['splits'][name]=metrics
            for k,v in rs.items():raw[f'{seed}__{name}__{k}']=v
            print('SCORE',seed,name,metrics['full_exact'],'/',metrics['n'],'edge',metrics['edge_exact'],'node',metrics['node_exact'],'pairs',metrics['exact_pairs'],flush=True)
    (out/'E35_METRICS.json').write_text(json.dumps(allmetrics,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'E35_RAW_PREDICTIONS.npz',**raw)
if __name__=='__main__':main()

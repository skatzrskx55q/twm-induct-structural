#!/usr/bin/env python3
"""TWM post-E36 research prototype; NO new promoted experimental stage.
Shared learned event token encoder plus differentiable episodic ridge edge/node
heads.  Input only 8 observed transitions and query state/action. Law identifiers
never enter the neural model. CPU implementation, n>=2, homogeneous episode n.

Training targets are disjoint from features. Query-after states are never read.
The hard edge mask uses a straight-through estimator for joint node gradients.
"""
import os, json, time, argparse, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
UPSTREAM=HERE.parent/'upstream'
import sys
sys.path.insert(0,str(UPSTREAM))
from e34_induction import local_features,node_features

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)

def encode_rows(rows):
    """NO future labels in model inputs; query targets returned only for loss/eval."""
    fields=['es','eq','ey','et','ns','ny','nt','sn','sb','sa','qn','qb','qa']
    data={k:[] for k in fields}; meta=[]
    for r in rows:
        n=r['n_nodes']; assert n>=2 and len(r['history'])==8
        offdiag=~np.eye(n,dtype=bool)
        es=[];ey=[];ns=[];ny=[];snode=[];sedge=[];sact=[]
        for t in r['history']:
            eb=np.asarray(t['edges_before'],dtype=np.float32).reshape(n,n)
            ea=np.asarray(t['edges_after'],dtype=np.float32).reshape(n,n)
            before=np.asarray(t['nodes_before'],dtype=np.float32)
            e,_=local_features(before,eb,t['action'])
            es.append(e[offdiag]);ey.append(np.bitwise_xor(eb.astype('i1'),ea.astype('i1'))[offdiag]);
            ns.append(node_features(before,eb,ea,t['action']))
            ny.append(np.bitwise_xor(before.astype('i1'),np.asarray(t['nodes_after'],dtype='i1')))
            snode.append(before);sedge.append(eb);sact.append(t['action'])
        querynodes=np.asarray(r['query_nodes'],dtype='f4');queryedges=np.asarray(r['query_edges'],dtype='f4').reshape(n,n)
        e,_=local_features(querynodes,queryedges,r['future_actions'][0])
        futureedges=np.asarray(r.get('future_edges',r['query_edges']),dtype='i1').reshape(n,n)
        future_nodes=np.asarray(r.get('future_nodes',r['query_nodes']),dtype='i1')
        values={
            'es':np.concatenate(es).astype('f4'), 'eq':e[offdiag].astype('f4'),
            'ey':np.concatenate(ey).astype('f4'),'et':np.bitwise_xor(queryedges.astype('i1'),futureedges)[offdiag].astype('f4'),
            'ns':np.concatenate(ns).astype('f4'),'ny':np.concatenate(ny).astype('f4'),
            'nt':np.bitwise_xor(querynodes.astype('i1'),future_nodes).astype('f4'),
            'sn':np.asarray(snode,dtype='f4'),'sb':np.asarray(sedge,dtype='f4'),'sa':np.asarray(sact,dtype='i8'),
            'qn':querynodes,'qb':queryedges,'qa':np.int64(r['future_actions'][0])}
        for k in fields:data[k].append(values[k])
        meta.append([str(r.get('law_id','unknown')),str(r.get('pair_id','')),str(r.get('history_hash',''))])
    return {k:torch.from_numpy(np.asarray(v)) for k,v in data.items()}|{'meta':meta}

def batch(data, ix):
    return {k:v[ix] for k,v in data.items() if k!='meta'}

def node_features_torch(nodes, edges, action):
    """Tensor derivative of E34's generic node_features; no special law semantics."""
    B,n=nodes.shape
    den=max(1,n-1)
    act=F.one_hot(action,num_classes=n).to(nodes.dtype)
    def one(A):
        # row is source, column is target, consistent with historical E34
        A2=torch.bmm(A,A)/den
        indeg=A.sum(dim=1)/den;outdeg=A.sum(dim=2)/den
        arow=torch.einsum('bi,bij->bj',act,A)
        acol=torch.einsum('bij,bj->bi',A,act)
        x_in=torch.bmm(A.transpose(1,2),nodes.unsqueeze(-1)).squeeze(-1)/den
        x_out=torch.bmm(A,nodes.unsqueeze(-1)).squeeze(-1)/den
        a2row=torch.einsum('bi,bij->bj',act,A2)
        actor_val=(act*nodes).sum(-1,keepdim=True).expand(-1,n)
        return torch.stack((nodes,act,arow,acol,x_in,x_out,a2row,actor_val,indeg,outdeg),-1)
    return one(edges)

def build_node_query(nodes, edge_before, edge_flip_prob, action):
    B,n=nodes.shape
    diag=torch.eye(n,device=nodes.device,dtype=torch.bool)
    offdiag=(~diag)
    # ST hard mask: real binary world state in forward, relaxed gradients in backward
    hard=(edge_flip_prob>=.5).to(edge_flip_prob.dtype)
    straight=hard.detach()+edge_flip_prob-edge_flip_prob.detach()
    mask=torch.zeros((B,n,n),device=nodes.device,dtype=nodes.dtype)
    mask[:,offdiag]=straight
    after=edge_before+(1-2*edge_before)*mask
    old=node_features_torch(nodes,edge_before,action)
    new=node_features_torch(nodes,after,action)
    return torch.cat((old,new,new-old),-1),mask,after

class Ridge(nn.Module):
    def __init__(self,d,extra,hid=24):
        super().__init__()
        idx,jdx=torch.triu_indices(d,d,1)
        self.register_buffer('idx',idx);self.register_buffer('jdx',jdx)
        self.feat=nn.Sequential(nn.Linear(d+len(idx)+extra,72),nn.Tanh(),nn.Linear(72,hid),nn.Tanh())
        self.log_lambda=nn.Parameter(torch.tensor(-.3))
        self.temperature=nn.Parameter(torch.tensor(4.0))
    def embed(self,x,extra):
        poly=x[...,self.idx]*x[...,self.jdx]
        h=self.feat(torch.cat((x,poly,extra),dim=-1))
        return torch.cat((torch.ones_like(h[...,:1]),h),-1)
    def predict(self,X,Q,labels):
        dim=X.shape[-1];xt=X.transpose(1,2)
        lam=F.softplus(self.log_lambda)+.02
        eye=torch.eye(dim,dtype=X.dtype,device=X.device)[None,:,:]
        weights=torch.linalg.solve(xt@X+lam*eye,xt@labels[...,None])
        raw=(Q@weights).squeeze(-1)
        return torch.sigmoid((raw-.5)*self.temperature.clamp(.5,12.))

class JointEventModel(nn.Module):
    def __init__(self,capacity=8,detach_node_events=False):
        super().__init__();self.capacity=capacity;self.detach_node_events=detach_node_events
        self.event=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,capacity),nn.Tanh())
        self.edge=Ridge(12,capacity)
        self.node=Ridge(30,4*capacity)
    @staticmethod
    def node_context(event,flip,n):
        """4 permutation equivariant pooled blocks for each target/source node."""
        B,T,p,d=event.shape
        offdiag=~torch.eye(n,device=event.device,dtype=torch.bool)
        full=torch.zeros((B,T,n,n,d),dtype=event.dtype,device=event.device)
        mask=torch.zeros((B,T,n,n),dtype=event.dtype,device=event.device)
        full[:,:,offdiag,:]=event
        mask[:,:,offdiag]=flip.reshape(B,T,p)
        ins=full.sum(dim=2)/max(1,n-1)
        outs=full.sum(dim=3)/max(1,n-1)
        ins_event=(full*mask[...,None]).sum(dim=2)/max(1,n-1)
        outs_event=(full*mask[...,None]).sum(dim=3)/max(1,n-1)
        return torch.cat((ins,outs,ins_event,outs_event),dim=-1)
    def forward(self,b):
        B,n=b['qn'].shape
        p=n*(n-1);s=b['es'].shape[1]
        assert s==8*p, (s,p)
        edge_sup=self.event(b['es']);edge_query=self.event(b['eq'])
        xe=self.edge.embed(b['es'],edge_sup)
        qe=self.edge.embed(b['eq'],edge_query)
        pred_edge=self.edge.predict(xe,qe,b['ey'])
        node_edge=pred_edge.detach() if self.detach_node_events else pred_edge
        event_sup=edge_sup.detach() if self.detach_node_events else edge_sup
        event_query=edge_query.detach() if self.detach_node_events else edge_query
        node_query,hardmask,after=build_node_query(b['qn'],b['qb'],node_edge,b['qa'])
        support_context=self.node_context(event_sup.reshape(B,8,p,-1),b['ey'].reshape(B,8,p),n).reshape(B,8*n,-1)
        query_context=self.node_context(event_query[:,None],node_edge[:,None],n)[:,0]
        xn=self.node.embed(b['ns'],support_context)
        qn=self.node.embed(node_query,query_context)
        pred_node=self.node.predict(xn,qn,b['ny'])
        return pred_edge,pred_node

def train(model, data, steps, seed, batch_size=16, lr=.0012, log_every=100, node_weight=1.):
    seed_all(seed)
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    model.train();records=[];start=time.monotonic()
    for t in range(steps):
        ids=torch.from_numpy(rng.integers(len(data['qn']),size=batch_size,dtype=np.int64))
        b=batch(data,ids)
        ep,npred=model(b)
        le=F.binary_cross_entropy(ep.clamp(1e-6,1-1e-6),b['et'])
        ln=F.binary_cross_entropy(npred.clamp(1e-6,1-1e-6),b['nt'])
        loss=le+node_weight*ln
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if t==0 or (t+1)%log_every==0 or t==steps-1:
            row={'step':t+1,'edge_bce':round(le.item(),6),'node_bce':round(ln.item(),6),'seconds':round(time.monotonic()-start,1)}
            records.append(row);print('TRAIN',seed,row,flush=True)
    return opt.state_dict(),records

@torch.no_grad()
def evaluate(model,data,batch_size=64):
    model.eval();pr_e=[];pr_n=[]
    for st in range(0,len(data['qn']),batch_size):
        b=batch(data,slice(st,st+batch_size))
        pe,pn=model(b)
        pr_e.append(pe.cpu());pr_n.append(pn.cpu())
    pe=torch.cat(pr_e);pn=torch.cat(pr_n)
    e=((pe>=.5)==data['et'].bool()).all(-1)
    n=((pn>=.5)==data['nt'].bool()).all(-1)
    f=e&n
    laws={}
    for ok,(law,*_) in zip(f.tolist(),data['meta']):
        z=laws.setdefault(law,{'n':0,'full':0});z['n']+=1;z['full']+=int(ok)
    pairs=None
    if len(f)%2==0 and all(data['meta'][i][1] and data['meta'][i][1]==data['meta'][i+1][1] for i in range(0,len(f),2)):
        pairs=int(sum(bool(f[i]) and bool(f[i+1]) for i in range(0,len(f),2)))
    return {'n':len(f),'exact_pairs':pairs,'full_exact':int(f.sum()),'edge_exact':int(e.sum()),'node_exact':int(n.sum()),'node_only_errors':int((e&~n).sum()),'edge_only_errors':int((~e&n).sum()),'both_errors':int((~e&~n).sum()),'per_law':laws},(pe,pn)

def permute_row(r,perm):
    """Permutation is new-index -> old-index. Targets are permuted only for verification."""
    q=dict(r);n=r['n_nodes'];perm=np.asarray(perm);inv=np.argsort(perm)
    def remap_nodes(nodes):return np.asarray(nodes)[perm].tolist()
    def remap_edges(edges):return np.asarray(edges).reshape(n,n)[np.ix_(perm,perm)].reshape(-1).tolist()
    def remap_action(action):return int(inv[action])
    q['history']=[]
    for t in r['history']:
        s=dict(t)
        for f in ('nodes_before','nodes_after'):s[f]=remap_nodes(t[f])
        for f in ('edges_before','edges_after'):s[f]=remap_edges(t[f])
        s['action']=remap_action(t['action']);q['history'].append(s)
    for f in ('query_nodes','future_nodes'):
        if f in r:q[f]=remap_nodes(r[f])
    for f in ('query_edges','future_edges'):
        if f in r:q[f]=remap_edges(r[f])
    q['future_actions']=[remap_action(a) for a in r['future_actions']]
    return q

def self_tests(rows,model,check_count=3):
    """Gradient, numeric feature parity, label isolation, and permutation checks."""
    from e34_induction import node_features
    r=rows[0];n=r['n_nodes'];x=np.asarray(r['query_nodes'],dtype='f4');b=np.asarray(r['query_edges'],dtype='f4').reshape(n,n)
    a=r['future_actions'][0]
    f0=node_features(x,b,b,a)
    f1,_,_=build_node_query(torch.tensor(x)[None],torch.tensor(b)[None],torch.zeros((1,n*(n-1))),torch.tensor([a]))
    parity=float(np.max(np.abs(f0-f1[0].detach().numpy())))
    assert parity<2e-6, ('feature_parity',parity)
    # No dependence of output on query targets; et / nt ONLY in loss
    pack0=encode_rows(rows[:2]);part=batch(pack0,slice(None));model.eval()
    with torch.no_grad():
        p0=model(part)
        part['et']=1-part['et'];part['nt']=1-part['nt']
        p1=model(part)
    leak=max(float(torch.max(torch.abs(y-z))) for y,z in zip(p0,p1));assert leak==0, ('future leakage',leak)
    # Law metadata must never reach the network
    mcopy=dict(pack0);mcopy['meta']=[('ADVERSARIAL_SHUFFLED_LAW','unrelated','x')]*2
    with torch.no_grad():probe=model(batch(mcopy,slice(None)))
    meta_leak=max(float((y-z).abs().max()) for y,z in zip(p0,probe))
    assert meta_leak==0, ('law-id leakage',meta_leak)
    # Permute entity IDs everywhere in context and query; prediction should commute
    permdiff=0.
    rng=np.random.default_rng(12345)
    for row in rows[:check_count]:
        n=row['n_nodes'];perm=rng.permutation(n)
        original=encode_rows([row]);permuted=encode_rows([permute_row(row,perm)])
        with torch.no_grad():
            pe,pn=model(batch(original,slice(None)));qe,qn=model(batch(permuted,slice(None)))
        eo=torch.zeros((n,n));eo[~torch.eye(n,dtype=torch.bool)]=pe[0]
        eq=torch.zeros((n,n));eq[~torch.eye(n,dtype=torch.bool)]=qe[0]
        permdiff=max(permdiff,float((eq-eo[perm][:,perm]).abs().max()),float((qn[0]-pn[0][perm]).abs().max()))
    assert permdiff<1e-5, ('permutation error',permdiff)
    # Both losses should update shared event encoder (independently of edge loss)
    model.train();b=batch(encode_rows(rows[:3]),slice(None))
    model.zero_grad();a,c=model(b);F.binary_cross_entropy(c,b['nt']).backward()
    node_to_event=float(sum(z.grad.abs().sum().item() for z in model.event.parameters() if z.grad is not None))
    assert (node_to_event<1e-12 if model.detach_node_events else node_to_event>1e-8), ('node event coupling',node_to_event)
    return {'feature_parity_max_error':parity,'future_label_leakage_max_error':leak,'law_id_leakage_max_error':meta_leak,'permutation_max_error':permdiff,'node_gradient_to_shared_encoder_L1':node_to_event}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default=str(HERE.parent/'data/train_records.json'))
    p.add_argument('--validation',default=str(HERE.parent/'data/known_val_records.json'))
    p.add_argument('--output',default=str(HERE.parent/'results'))
    p.add_argument('--seeds',default='17')
    p.add_argument('--steps',type=int,default=250)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=.0012)
    p.add_argument('--test-only',action='store_true')
    p.add_argument('--ablation',choices=['none','stop_node_gradient'],default='none')
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    print('ENCODE DATA',flush=True)
    rows_train=json.loads(Path(args.train).read_text())['rows']
    rows_val=json.loads(Path(args.validation).read_text())['rows']
    dtrain=encode_rows(rows_train);dval=encode_rows(rows_val)
    summary={'status':'EXPLORATORY_OPEN_DATA_ONLY_NO_SEALED_TEST','max_steps':args.steps,'datasets':{'train':len(rows_train),'validation':len(rows_val)},'seeds':{},'model':'SharedEventMetaRidge-PILOT', 'test_only':args.test_only,'ablation':args.ablation}
    for s in map(int,args.seeds.split(',')):
        seed_all(s);model=JointEventModel(detach_node_events=(args.ablation=='stop_node_gradient'))
        params=sum(v.numel() for v in model.parameters());summary['parameter_count']=params
        print('MODEL',s,params,flush=True)
        st=time.monotonic()
        tests=self_tests(rows_val,model)
        if args.test_only:
            summary['seeds'][str(s)]={'self_tests_untrained':tests}
            continue
        opt,logs=train(model,dtrain,args.steps,s,batch_size=args.batch,lr=args.lr)
        train_metrics,_=evaluate(model,dtrain)
        val_metrics,_=evaluate(model,dval)
        aftertests=self_tests(rows_val,model)
        summary['seeds'][str(s)]={'logs':logs,'tests_before':tests,'tests_after':aftertests,'train':train_metrics,'known_val':val_metrics,'seconds_total':round(time.monotonic()-st,1)}
        torch.save({'seed':s,'steps':args.steps,'model':model.state_dict(),'optimizer':opt,'ablation':args.ablation,'status':'PILOT_NOT_PROMOTED'},out/f'joint_event_seed{s}_pilot.pt')
        print('RESULT',s,'train',train_metrics['full_exact'],len(rows_train),'val',val_metrics['full_exact'],len(rows_val),flush=True)
    (out/'pilot_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

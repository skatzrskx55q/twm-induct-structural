"""TWM LawBench v0 reference evaluator. Research oracle only: NEVER import in a neural runtime.
Registered specification: ../N2_PROTOCOL_PRE_REGISTERED.json
State is (list of node bits, n*n directed edge matrix flattened, diagonal zeros).
"""
from __future__ import annotations
import numpy as np

EDGE_ATOMS=('edge','reverse_edge','src_on','dst_on','src_actor','dst_actor','twohop','actor_to_src','dst_to_actor')
NODE_ATOMS=('self_on','is_actor','from_actor','to_actor','incoming_on','outgoing_on','twohop_actor','actor_on')
OPS=('and','or','xor')

def read_edges(edges,n):
    a=np.asarray(edges,dtype=np.uint8).reshape(n,n)
    assert a.shape==(n,n) and (np.diag(a)==0).all() and np.isin(a,[0,1]).all()
    return a

def edge_atoms(nodes,edges,actor,n):
    x=np.asarray(nodes,dtype=np.uint8);a=read_edges(edges,n)
    i,j=np.indices((n,n)); positive=lambda z:z.astype(bool)
    z={
        'edge':positive(a), 'reverse_edge':positive(a.T),
        'src_on':positive(x[i]), 'dst_on':positive(x[j]),
        'src_actor':i==actor, 'dst_actor':j==actor,
        'twohop':positive((a.astype(np.int16)@a.astype(np.int16))>0),
        'actor_to_src':positive(a[actor,i]), 'dst_to_actor':positive(a[j,actor])
    }
    z['edge_threehop']=positive((a.astype(np.int16)@a.astype(np.int16)@a.astype(np.int16))>0)
    return z

def node_atoms(nodes,edges,actor,n):
    x=np.asarray(nodes,dtype=np.uint8);a=read_edges(edges,n)
    neighbors=(a.astype(np.int16)@x.astype(np.int16));incoming=(a.T.astype(np.int16)@x.astype(np.int16));reachable=(a.astype(np.int16)@a.astype(np.int16))
    z={
        'self_on':x.astype(bool), 'is_actor':np.arange(n)==actor,
        'from_actor':a[actor,:].astype(bool),'to_actor':a[:,actor].astype(bool),
        'incoming_on':(incoming>0), 'outgoing_on':(neighbors>0),
        'twohop_actor':(reachable[actor,:]>0),
        'actor_on':np.full(n,x[actor]>0,dtype=bool),
        'node_two_or_more_incoming_on':incoming>=2
    }
    return z

def parse(expr):
    if isinstance(expr,dict):return expr
    raise TypeError('Expression must be a frozen dict with p/q/op')

def apply_op(a,b,op):
    if op=='and':return np.logical_and(a,b)
    if op=='or':return np.logical_or(a,b)
    if op=='xor':return np.logical_xor(a,b)
    raise ValueError(op)

def edge_mask(nodes,edges,actor,n,expr):
    z=edge_atoms(nodes,edges,actor,n);r=apply_op(z[expr['p']],z[expr['q']],expr['op']).astype(np.uint8)
    np.fill_diagonal(r,0)
    return r

def node_mask(nodes,edges,actor,n,expr):
    z=node_atoms(nodes,edges,actor,n)
    return apply_op(z[expr['p']],z[expr['q']],expr['op']).astype(np.uint8)

def transition(nodes,edges,actor,n,law):
    assert len(nodes)==n and len(edges)==n*n and 0<=actor<n
    edge_after=(read_edges(edges,n)^edge_mask(nodes,edges,actor,n,law['edge'])).reshape(-1)
    node_after=np.asarray(nodes,dtype=np.uint8)^node_mask(nodes,edge_after,actor,n,law['node'])
    return node_after.astype(int).tolist(),edge_after.astype(int).tolist()

def signature(expr,kind,probes):
    bits=[]
    for nodes,edges,actor,n in probes:
        x=edge_mask(nodes,edges,actor,n,expr) if kind=='edge' else node_mask(nodes,edges,actor,n,expr)
        bits.extend(int(y) for y in x.flat)
    return bytes(bits)

def compose(p,op,q):return {'p':p,'op':op,'q':q}

def mechanism_key(m):return m['p']+'_'+m['op']+'_'+m['q']

def law_key(l):return mechanism_key(l['edge'])+'::'+mechanism_key(l['node'])

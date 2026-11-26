#!/usr/bin/env python3
"""E34: online contextual system identification with generic CART (not law-grammar search).
Requires only observed before/action/after transitions and current before/action.
No law IDs, law formulas, future labels, or simulator are used by `infer`.
No offline training; context-specific edge and node decision trees are refit each episode.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
MAX_DEPTH_EDGE=6; MAX_DEPTH_NODE=6; MIN_LEAF=1

def local_features(nodes, edges, action):
    n=len(nodes);x=np.asarray(nodes,dtype=np.float32)
    adj=np.asarray(edges,dtype=np.float32).reshape(n,n)
    i,j=np.indices((n,n));actor_on=float(x[action])
    two=(adj@adj)/max(1,n-1)
    indeg=adj.sum(0)/max(1,n-1);outdeg=adj.sum(1)/max(1,n-1)
    edge=np.stack((adj,adj.T,x[i],x[j],(i==action).astype('f4'),(j==action).astype('f4'),two,adj[action,i],adj[j,action],np.full((n,n),actor_on,dtype='f4'),np.broadcast_to(outdeg[:,None],(n,n)),np.broadcast_to(indeg[None,:],(n,n))),axis=-1).astype('f4')
    node=np.stack((x,(np.arange(n)==action).astype('f4'),adj[action,:],adj[:,action],(adj.T@x)/max(1,n-1),(adj@x)/max(1,n-1),two[action,:],np.full(n,actor_on,dtype='f4'),indeg,outdeg),axis=-1).astype('f4')
    return edge,node

def node_features(nodes,before_edges,after_edges,action):
    _,before=local_features(nodes,before_edges,action)
    _,after=local_features(nodes,after_edges,action)
    return np.concatenate((before,after,after-before),axis=-1)

def fit_predict(x,y,q,depth,seed):
    y=np.asarray(y,dtype=np.int8).reshape(-1)
    if np.all(y==y[0]):return np.full(len(q),int(y[0]),dtype=np.int8)
    tree=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=MIN_LEAF,random_state=seed)
    tree.fit(x,y)
    return tree.predict(q).astype(np.int8)

def infer(history, current_nodes, current_edges, actor, seed=17):
    """Predict changes as Boolean masks. Zero oracle access.
    Caller never passes labels for the current query.
    """
    n=len(current_nodes);diag=~np.eye(n,dtype=bool)
    ex=[];ey=[];nx=[];ny=[]
    for transition in history:
        a=transition['action']; nb=transition['nodes_before'];eb=transition['edges_before'];ea=transition['edges_after'];na=transition['nodes_after']
        e,_=local_features(nb,eb,a)
        ex.append(e[diag])
        ey.append((np.asarray(eb,dtype=np.int8).reshape(n,n)^np.asarray(ea,dtype=np.int8).reshape(n,n))[diag])
        nx.append(node_features(nb,eb,ea,a))
        ny.append(np.asarray(nb,dtype=np.int8)^np.asarray(na,dtype=np.int8))
    e_query,_=local_features(current_nodes,current_edges,actor)
    pred_edge=fit_predict(np.concatenate(ex),np.concatenate(ey),e_query[diag],MAX_DEPTH_EDGE,seed)
    after_edges=np.asarray(current_edges,dtype=np.int8).reshape(n,n).copy()
    after_edges[diag]^=pred_edge
    qnode=node_features(current_nodes,current_edges,after_edges,actor)
    pred_node=fit_predict(np.concatenate(nx),np.concatenate(ny),qnode,MAX_DEPTH_NODE,seed+10000)
    edge_mask=np.zeros((n,n),dtype=np.int8);edge_mask[diag]=pred_edge
    return edge_mask,pred_node

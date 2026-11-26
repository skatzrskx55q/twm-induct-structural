"""TWM INDUCT research pilot: learned episodic conditional partitions.

NO hardcoded law atoms, operations, rule IDs, or law oracle in inference.
Shared learned soft pair-of-predicate gates create four local regions per tree.
Each episode fits independent region probabilities from observed support transitions.
An optional learned *support-conditioned* weighting selects trees using leave-one-out
predictive loss, not oracle query labels. This is not a CART executor.

The baseline 'static' uses identical learned gates, trainable params, region fitting
and optimizer budget but no per-episode selection by support fit.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class StructuralNode(nn.Module):
    def __init__(self, d=30, trees=24, conditioned=True, prior=0.7):
        super().__init__()
        self.d=d;self.trees=trees;self.conditioned=conditioned;self.prior=prior
        self.predicates=nn.Linear(d,2*trees)
        nn.init.xavier_uniform_(self.predicates.weight, gain=0.65)
        nn.init.zeros_(self.predicates.bias)
        self.log_gate_scale=nn.Parameter(torch.tensor(1.1))
        self.log_reliability=nn.Parameter(torch.tensor(1.0))
        self.tree_logits=nn.Parameter(torch.zeros(trees))
        self.logit_temperature=nn.Parameter(torch.tensor(1.5))

    def membership(self,x):
        b,s,d=x.shape
        gate=torch.sigmoid(self.predicates(x).reshape(b,s,self.trees,2)*self.log_gate_scale.exp().clamp(1.0,12.0))
        a,c=gate.unbind(-1)
        # Every soft tree has four learned conjunctive leaves.
        h=torch.stack((a*c,a*(1-c),(1-a)*c,(1-a)*(1-c)),dim=-1)
        return h

    def make_context(self,s,sy):
        h=self.membership(s) # [B,S,M,4]
        counts=h.sum(1)
        pos=torch.einsum('bsmc,bs->bmc',h,sy)
        prob=(pos+self.prior*0.5)/(counts+self.prior)
        # Episode-specific reliability from support leave-one-out predictions.
        # Self evidence is subtracted, including its fractional membership.
        loo_num=pos[:,None,:,:]-h*sy[:,:,None,None]+self.prior*0.5
        loo_den=counts[:,None,:,:]-h+self.prior
        loo_leaf=loo_num/loo_den.clamp_min(1e-5)
        loo_p=(h*loo_leaf).sum(-1).clamp(1e-5,1-1e-5)
        y=sy[:,:,None]
        error=-(y*loo_p.log()+(1-y)*(1-loo_p).log()).mean(dim=1)
        if self.conditioned:
            strength=self.log_reliability.exp().clamp(0.05,40.0)
            weights=F.softmax(self.tree_logits[None,:]-strength*error,dim=-1)
        else:
            weights=F.softmax(self.tree_logits[None,:].expand_as(error),dim=-1)
        return prob,weights,error

    def predict_context(self,query,context):
        leaf,weights,_=context
        q=self.membership(query)
        trees=(q*leaf[:,None,:,:]).sum(-1)
        raw=(trees*weights[:,None,:]).sum(-1)
        # Temperature changes decision margins but not whether raw >= 0.5.
        logits=torch.logit(raw.clamp(1e-5,1-1e-5))
        out=torch.sigmoid(logits*self.logit_temperature.exp().clamp(.3,12.))
        return out

    def forward(self,s,query,sy):
        return self.predict_context(query,self.make_context(s,sy))

    def diagnostics(self,s,sy):
        with torch.no_grad():
            _,w,errors=self.make_context(s,sy)
            entropy=-(w*w.clamp_min(1e-9).log()).sum(-1)
            return {'entropy':entropy.mean().item(),
                    'effective_trees':w.exp().mean().item() if False else entropy.exp().mean().item(),
                    'max_weight':w.max(-1).values.mean().item(),
                    'spread':errors.std(-1).mean().item()}

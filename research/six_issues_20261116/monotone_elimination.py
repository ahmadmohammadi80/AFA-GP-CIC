"""Monotone confidence-safe policy elimination with dynamic influence support.

Research candidate, not a claim of globally new successive-elimination theory.
For fixed Q,H and anytime-valid pairwise confidence intervals, intersect each
new interval with its previous one. Permanently eliminate q when U(q) <
max_p L(p). Restrict FUTURE label requests to edges influencing surviving
policies, while preserving previously collected confidence information.

A subtlety: the witness p attaining max L can itself later be eliminated.
Nevertheless max L over ALL original candidates is monotone, so store that
global incumbent lower bound; the best true candidate can never be removed
on the joint coverage event. Never use an upper confidence bound of the
INCUMBENT ONLY as a regret certificate.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations

def make_coefficients(b,Q,H):
    b=np.asarray(b,float);Q=np.asarray(Q,float);H=np.asarray(H,float)
    k=len(b)
    if Q.ndim!=2 or H.ndim!=2 or Q.shape[1]!=k or H.shape[1]!=k:
        raise ValueError("dimensions")
    if not np.allclose(b.sum(),1) or not np.allclose(Q.sum(axis=1),1) or not np.allclose(H.sum(axis=1),1):
        raise ValueError("probability simplex")
    edges=list(combinations(range(k),2))
    A=np.array([[[(q[i]-b[i])*h[j]-(q[j]-b[j])*h[i] for i,j in edges] for h in H] for q in Q])
    return A,edges

class MonotoneEliminator:
    def __init__(self,A):
        A=np.asarray(A,float)
        if A.ndim!=3 or not all(A.shape) or not np.all(np.isfinite(A)):
            raise ValueError("A shape")
        self.A=A.copy();self.lo=np.full(A.shape[-1],-1.)
        self.hi=np.full(A.shape[-1],1.)
        self.alive=np.ones(A.shape[0],bool)
        self.incumbent_lower=-np.inf
        self.count_updates=0
        self.history=[]

    def bounds(self):
        a=self.A
        lower=np.where(a>=0,a*self.lo,a*self.hi).sum(axis=2).min(axis=1)
        upper=np.where(a>=0,a*self.hi,a*self.lo).sum(axis=2).min(axis=1)
        return lower,upper

    def update(self,e,newlo,newhi):
        if not isinstance(e,(int,np.integer)) or e<0 or e>=len(self.lo):
            raise ValueError("edge")
        if not np.isfinite(newlo) or not np.isfinite(newhi) or newlo>newhi or newlo< -1 or newhi>1:
            raise ValueError("confidence interval")
        # Intersecting valid CS sets remains valid on their simultaneous
        # coverage event, even when individual CS endpoints are not nested.
        lo=max(float(newlo),self.lo[e])
        hi=min(float(newhi),self.hi[e])
        if lo>hi:
            # A conflict implies at least one confidence event failed.
            # Revert to the prior set to avoid claiming an empty certificate.
            self.history.append({"edge":int(e),"conflict":True})
            return False
        self.lo[e]=lo;self.hi[e]=hi
        L,U=self.bounds()
        self.incumbent_lower=max(self.incumbent_lower,float(np.max(L)))
        # Exact comparisons; equality is retained.
        self.alive &= U>=self.incumbent_lower
        self.count_updates+=1
        self.history.append({"edge":int(e),"survivors":int(self.alive.sum()),
                             "active_edges":int(self.active_edges().sum())})
        return True

    def active_edges(self):
        # Union over surviving q AND all h. If all eliminated due to a
        # coverage failure, report no active edges and do not deploy.
        if not np.any(self.alive):
            return np.zeros(self.A.shape[-1],bool)
        return np.any(self.A[self.alive]!=0,axis=(0,1))

    def result(self):
        L,U=self.bounds()
        if not self.alive.any():
            return {"valid_candidate":False,"survivors":[],"active_edges":[]}
        idx=int(np.argmax(np.where(self.alive,L,-np.inf)))
        # Best original candidate may have been removed only on failure
        # of coverage; keep full upper for a valid conservative regret gap.
        return {"valid_candidate":True,"selected":idx,
                "lower":float(L[idx]),"upper_best":float(np.max(U)),
                "simple_regret_upper":float(max(0.,np.max(U)-L[idx])),
                "survivors":np.flatnonzero(self.alive).tolist(),
                "active_edges":np.flatnonzero(self.active_edges()).tolist()}

def witness():
    # Four policies with disjoint relevant edges. q0 has strong positive
    # true gain, q1 is provably negative, q2 remains ambiguous.
    A=np.zeros((3,1,4))
    A[0,0,0]=.2;A[1,0,1]=.2;A[2,0,2]=.2
    state=MonotoneEliminator(A)
    state.update(0,.8,.9)
    state.update(1,-.9,-.8)
    state.update(2,-.1,.1)
    assert state.alive.tolist()==[True,False,False]
    return {"initial_edges":4,"final_edges":state.result()["active_edges"],
            "survivors":state.result()["survivors"],
            "result":state.result()}

if __name__=="__main__":
    import json
    print(json.dumps(witness(),indent=2))

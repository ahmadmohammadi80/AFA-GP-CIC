"""Certificate-influence support pruning: a precise, restricted algorithmic claim.

For finite fixed Q,H and baseline b, A[q,h,e] is the signed coefficient of
edge e in gain(q,h). An edge is globally irrelevant if ALL its coefficients
vanish. It cannot affect ANY candidate's lower certificate or true gain.

Operator: query only edges in the union support of A, but retain a common
anytime-valid confidence budget delta / |support|. This is *not* claimed
globally novel; it is a mathematically exact reduction with an explicit
uniform-query sample-efficiency separation in sparse instances.
"""
import math
from itertools import combinations
import numpy as np
from scipy.stats import binomtest

def coefficients(b,Q,H):
    b=np.asarray(b,float);Q=np.asarray(Q,float);H=np.asarray(H,float)
    k=len(b)
    if Q.ndim!=2 or H.ndim!=2 or Q.shape[1]!=k or H.shape[1]!=k:
        raise ValueError("shape")
    edges=list(combinations(range(k),2))
    d=Q-b
    A=np.array([[[v[i]*h[j]-v[j]*h[i] for i,j in edges]
                 for h in H] for v in d])
    return edges,A

def support(A):
    A=np.asarray(A)
    if A.ndim!=3:raise ValueError("tensor")
    return np.flatnonzero(np.any(A!=0,axis=(0,1)))

def certificate(A,lo,hi):
    A=np.asarray(A,float)
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    return np.min(np.sum(np.where(A>=0,A*lo,A*hi),axis=-1),axis=-1)

def demonstration(seed=0,k=20,n=10000,delta=.05):
    if k<6 or n<1:raise ValueError("parameters")
    b=np.zeros(k);b[:2]=.5
    Q=np.array([b.copy()]);Q[0,0]+=.1;Q[0,1]-=.1
    H=np.array([np.eye(k)[2],np.eye(k)[3]])
    edges,A=coefficients(b,Q,H);active=support(A)
    assert len(active)==4
    rng=np.random.default_rng(seed)
    mu=np.zeros(len(edges))
    for e,(i,j) in enumerate(edges):
        if i==0 and j in (2,3):mu[e]=.9
        if i==1 and j in (2,3):mu[e]=-.9
    def simulate(select):
        counts=np.zeros(len(edges),dtype=int)
        wins=np.zeros(len(edges),dtype=int)
        # Pre-generate independent Bernoulli streams on each edge; query order
        # does not change their law.
        streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(edges))]
        for t in range(n):
            e=int(select[t%len(select)])
            wins[e]+=streams[e].binomial(1,(1+mu[e])/2)
            counts[e]+=1
        lo=np.full(len(edges),-1.);hi=np.full(len(edges),1.)
        # All-n Hoeffding: union over e and n; inactive edges are irrelevant.
        E=len(select)
        for e in select:
            nn=counts[e];m=2*wins[e]/nn-1
            r=math.sqrt(2*math.log(E*math.pi**2*nn*nn/(3*delta))/nn)
            lo[e]=max(-1,m-r);hi[e]=min(1,m+r)
        return {"certificate":float(certificate(A,lo,hi)[0]),
                "minimum_active_count":int(min(counts[active])),
                "queried_inactive":int(counts.sum()-counts[active].sum())}
    return {"k":k,"edges":len(edges),"active_edges":len(active),
            "uniform":simulate(np.arange(len(edges))),
            "support_only":simulate(active),
            "true_gain":float(np.min(A[0]@mu)),
            "deterministic_query_allocation_ratio":len(edges)/len(active)}

if __name__=="__main__":
    import json
    print(json.dumps(demonstration(),indent=2))

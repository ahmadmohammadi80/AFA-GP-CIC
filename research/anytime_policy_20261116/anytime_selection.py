"""Time-uniform, policy-reusable pairwise certificates.

A candidate policy and opponents can be chosen after seeing ALL data, and
sampling/stopping can be adaptive, because the event covers all edges and
sample counts. This is a valid engineering extension, not an original
concentration theorem or independently established global novelty.
"""
import numpy as np
from itertools import combinations

def radius(n,delta,E):
    """Anytime Hoeffding via sum_{n>=1} 1/n^2=pi^2/6."""
    if not (0<delta<1) or E<1:raise ValueError("invalid confidence")
    n=np.asarray(n,float)
    if np.any(n<1):raise ValueError("at least one sample per edge")
    return np.sqrt(2*np.log((E*np.pi**2*n*n)/(3*delta))/n)

def edge_coefficients(b,q,H):
    b=np.asarray(b,float);q=np.asarray(q,float);H=np.asarray(H,float)
    if H.ndim!=2 or H.shape[1]!=len(b):raise ValueError("opponents")
    if min(b)<0 or min(q)<0 or not np.isclose(b.sum(),1) or not np.isclose(q.sum(),1):
        raise ValueError("policies")
    edges=list(combinations(range(len(b)),2));d=q-b
    A=np.asarray([[d[i]*h[j]-d[j]*h[i] for i,j in edges] for h in H])
    return edges,A

def reusable_lower(b,q,H,means,counts,delta=.05):
    edges,A=edge_coefficients(b,q,H)
    means=np.asarray(means,float);counts=np.asarray(counts,int)
    if len(means)!=len(edges) or len(counts)!=len(edges):raise ValueError("edges")
    rad=radius(counts,delta,len(edges))
    return float(np.min(A@means-np.abs(A)@rad))

def sample_run(seed=0,k=5,delta=.05,max_rounds=1500,threshold=.015):
    """Sequential selection between two KL-feasible candidates with reused data."""
    rng=np.random.default_rng(seed)
    b=np.ones(k)/k
    q1=b.copy();q1[0]+=.08;q1[1]-=.08
    q2=b.copy();q2[2]+=.08;q2[3]-=.08
    Q=[q1,q2]
    kappa=.05
    KL=[float(np.sum(q*np.log(q/b))) for q in Q]
    assert max(KL)<kappa
    H=np.eye(k)
    potential=np.array([.45,-.35,.25,-.15,-.2])
    W=potential[:,None]-potential[None,:]
    for i,j in [(0,2),(2,3),(3,0)]:
        W[i,j]+=.04;W[j,i]-=.04
    edges=list(combinations(range(k),2));E=len(edges)
    true=np.array([W[i,j] for i,j in edges])
    N=np.ones(E,dtype=int);sums=np.array([2*rng.binomial(1,(1+v)/2)-1 for v in true],float)
    history=[]
    # Policy selection uses SAME observations as confirmation; valid by uniform coverage.
    for step in range(max_rounds):
        means=sums/N
        scores=[reusable_lower(b,q,H,means,N,delta) for q in Q]
        winner=int(np.argmax(scores))
        if scores[winner]>threshold:
            return {'certified':True,'selected':winner,'samples':int(N.sum()),
                    'steps':step,'lower':float(scores[winner]),'true_gain':float(min((q-b)@W@h for h in H for q in [Q[winner]])),
                    'KL':KL[winner],'threshold':threshold}
        # Choose edge with highest current certificate uncertainty for winner,
        # not an all-edge round-robin.
        _,A=edge_coefficients(b,Q[winner],H)
        active=np.max(abs(A),axis=0)
        weights=active*radius(N,delta,E)
        e=int(np.argmax(weights))
        sums[e]+=2*rng.binomial(1,(1+true[e])/2)-1
        N[e]+=1
    scores=[reusable_lower(b,q,H,sums/N,N,delta) for q in Q]
    winner=int(np.argmax(scores))
    return {'certified':False,'selected':winner,'samples':int(N.sum()),
            'steps':max_rounds,'lower':float(scores[winner]),
            'true_gain':float(min((Q[winner]-b)@W@h for h in H)),
            'KL':KL[winner],'threshold':threshold}

if __name__=="__main__":
    import json
    out=[sample_run(seed=s) for s in range(40)]
    print(json.dumps({'trials':len(out),'certified':sum(x['certified'] for x in out),
        'false_certificates':sum(x['certified'] and x['true_gain']<=x['threshold'] for x in out),
        'average_samples':float(np.mean([x['samples'] for x in out])),
        'rows':out},indent=2))

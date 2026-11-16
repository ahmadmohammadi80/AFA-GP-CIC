"""Anytime-valid beta-mixture confidence sequences for pairwise preferences.

Stationary independent Bernoulli comparison stream for each pair, with
adaptive edge choice, policy selection and stopping. This is NOT a claim of
globally original confidence-sequence methodology.
"""
from __future__ import annotations
import math
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln


@lru_cache(maxsize=500000)
def edge_cs(wins:int, n:int, alpha:float):
    """Invert the Beta(.5,.5) mixture likelihood martingale; confidence for 2p-1."""
    if not 0<alpha<1 or n<0 or not 0<=wins<=n:
        raise ValueError("invalid counts or confidence")
    if n==0:return -1.,1.
    s,t=wins,n-wins
    logfactor=betaln(s+.5,t+.5)-betaln(.5,.5)
    bound=math.log(1/alpha)
    phat=s/n
    def score(p):
        if p<=0:return math.inf if s>0 else float(logfactor)
        if p>=1:return math.inf if t>0 else float(logfactor)
        return float(logfactor-s*math.log(p)-t*math.log1p(-p)-bound)
    low=0. if s==0 else brentq(score,1e-300,phat,xtol=1e-14)
    high=1. if t==0 else brentq(score,phat,1-1e-15,xtol=1e-14)
    return 2*low-1,2*high-1


def edge_coefficients(b,q,H):
    b=np.asarray(b,float);q=np.asarray(q,float);H=np.asarray(H,float)
    if H.ndim!=2 or H.shape[1]!=len(b):raise ValueError("bad opponent matrix")
    if min(b)<0 or min(q)<0 or not np.isclose(b.sum(),1) or not np.isclose(q.sum(),1):
        raise ValueError("invalid policies")
    edges=list(combinations(range(len(b)),2));d=q-b
    A=np.asarray([[d[i]*h[j]-d[j]*h[i] for i,j in edges] for h in H])
    return edges,A


def hoeffding_radius(n,delta,E):
    if not 0<delta<1 or E<1:raise ValueError("invalid confidence")
    n=np.asarray(n,float)
    if np.any(n<1):raise ValueError("n must be >=1")
    return np.minimum(2.,np.sqrt(2*np.log((E*np.pi**2*n*n)/(3*delta))/n))


def certificate(A,lo,hi):
    A=np.asarray(A,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if A.ndim!=2 or A.shape[1]!=len(lo) or len(lo)!=len(hi):raise ValueError("dimensions")
    return float(np.min(np.where(A>=0,A*lo,A*hi).sum(axis=1)))


def run_one(seed=0,max_labels=1510,k=5,threshold=.015,delta=.05,mode="mixture",null=False):
    if mode not in ("mixture","hoeffding") or k!=5 or max_labels<10 or not 0<delta<1:
        raise ValueError("invalid setup")
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
    if null:W=np.zeros_like(W)
    edges=list(combinations(range(k),2));E=len(edges)
    # Counterfactual coupling: each algorithm sees the same random prefix
    # of each edge's stream despite selecting edges in a different order.
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(E)]
    true=np.array([W[i,j] for i,j in edges])
    N=np.ones(E,dtype=int)
    wins=np.array([streams[e].binomial(1,(1+t)/2) for e,t in enumerate(true)],int)
    AA=[edge_coefficients(b,q,H)[1] for q in Q]
    lo,hi=np.full(E,-1.,float),np.full(E,1.,float)
    breached=False
    def update(e):
        if mode=="mixture":
            lo[e],hi[e]=edge_cs(int(wins[e]),int(N[e]),delta/E)
        else:
            rad=float(hoeffding_radius(np.array([N[e]]),delta,E)[0])
            m=2*wins[e]/N[e]-1
            lo[e],hi[e]=max(-1.,m-rad),min(1.,m+rad)
    for e in range(E):update(e)
    breached=bool(np.any(lo>true+1e-12) or np.any(hi<true-1e-12))
    for step in range(max_labels-E+1):
        vals=np.array([certificate(A,lo,hi) for A in AA])
        winner=int(np.argmax(vals))
        if vals[winner]>threshold:
            return {'certified':True,'labels':int(N.sum()),'selected':winner,
                    'lower':float(vals[winner]),
                    'true_gain':float(min((Q[winner]-b)@W@h for h in H)),
                    'KL':KL[winner],'method':mode,'coverage_breached':breached}
        if N.sum()>=max_labels:break
        A=AA[winner];means=(lo+hi)/2
        worst=int(np.argmin(A@means-np.abs(A)@((hi-lo)/2)))
        priority=np.abs(A[worst])*(hi-lo)/(np.sqrt(N)+.2)
        e=int(np.argmax(priority))
        if priority[e]==0:e=int(np.argmin(N))
        wins[e]+=streams[e].binomial(1,(1+true[e])/2)
        N[e]+=1
        update(e)
        breached=bool(breached or lo[e]>true[e]+1e-12 or hi[e]<true[e]-1e-12)
    vals=np.array([certificate(A,lo,hi) for A in AA])
    winner=int(np.argmax(vals))
    return {'certified':False,'labels':int(N.sum()),'selected':winner,
            'lower':float(vals[winner]),
            'true_gain':float(min((Q[winner]-b)@W@h for h in H)),
            'KL':KL[winner],'method':mode,'coverage_breached':breached}


def experiment(seeds=40,max_labels=1510,threshold=.015):
    results={}
    for mode in ('mixture','hoeffding'):
        rows=[run_one(seed=20261116+s,max_labels=max_labels,threshold=threshold,mode=mode)
              for s in range(seeds)]
        results[mode]={'certified_count':sum(x['certified'] for x in rows),
           'false_certificates':sum(x['certified'] and x['true_gain']<=threshold for x in rows),
           'coverage_breaches':sum(x['coverage_breached'] for x in rows),
           'mean_labels':float(np.mean([x['labels'] for x in rows])),
           'mean_lower':float(np.mean([x['lower'] for x in rows])),
           'rows':rows}
    return {'seeds':seeds,'max_labels':max_labels,'threshold':threshold,'methods':results}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=40)
    p.add_argument('--max-labels',type=int,default=1510)
    p.add_argument('--threshold',type=float,default=.015)
    p.add_argument('--out',default='results/mixture_power.json')
    args=p.parse_args()
    data=experiment(args.seeds,args.max_labels,args.threshold)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in data['methods'].items()},indent=2))

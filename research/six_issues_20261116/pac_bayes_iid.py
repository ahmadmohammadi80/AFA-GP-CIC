"""PAC-Bayes *point-mass specialization* as an anytime-valid certification baseline.

Seeger (2002) PAC-Bayes-kl for iid bounded losses; the uniform prior on
FINITE (q,h) and posterior point masses yields complexity log(|Q||H|).
We combine it with a sum_{n>=1} 1/n^2 allocation and Pinsker inversion.
Under fixed uniform IID edge queries after a shared deterministic pilot,
each importance-weighted edge outcome is iid for every q,h. The pilot
is deliberately *not* used in the PAC-Bayes bound to preserve iid validity.
This is NOT general posterior PAC-Bayes optimization or an adaptive-edge
PAC-Bayes guarantee. It is a fair, if conservative, available baseline.

Reference: Seeger, "PAC-Bayesian Generalisation Error Bounds for Gaussian
Process Classification", JMLR 2002.
"""
from __future__ import annotations
import math,json,argparse
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import binomtest
from published_baselines import preference_problem

def radii(n,delta,count):
    if n<1 or not 0<delta<1 or count<1:raise ValueError("confidence")
    delta_n=6*delta/(math.pi**2*n*n)
    complexity=math.log((2*math.sqrt(n)*count)/delta_n)/n
    return min(1.,math.sqrt(complexity/2))

def run(seed,budget=1510,delta=.05,threshold=.015,cyclic=False):
    b,Q,H,W=preference_problem(cyclic)
    es=list(combinations(range(len(b)),2));E=len(es);R=len(Q)*len(H)
    A=np.array([[(q[i]-b[i])*h[j]-(q[j]-b[j])*h[i] for i,j in es]
                for q in Q for h in H])
    B=float(np.max(np.abs(A))*E)
    if B==0:raise ValueError("zero coefficients")
    mu=np.array([W[i,j] for i,j in es])
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(E)]
    edge_rng=np.random.default_rng(np.random.SeedSequence([seed,874343]))
    # Same pilot labels as the active methods, but the formal iid PAC-Bayes
    # certificate uses only subsequent independently uniform-selected edges.
    for e in range(E):streams[e].binomial(1,(1+mu[e])/2)
    sums=np.zeros(R);selection=None
    count=E;lastlo=np.full(R,-B)
    while count<budget:
        e=int(edge_rng.integers(0,E))
        y=2*int(streams[e].binomial(1,(1+mu[e])/2))-1
        sums+=(A[:,e]*E*y+B)/(2*B)
        count+=1;n=count-E
        em=sums/n
        rad=radii(n,delta,R)
        lastlo=2*B*np.maximum(0.,em-rad)-B
        g=lastlo.reshape(len(Q),len(H)).min(axis=1)
        if g.max()>threshold:
            selection=int(np.argmax(g));break
    true=[min(float((q-b)@W@h) for h in H) for q in Q]
    return {"seed":seed,"labels":count,"certified":selection is not None,
            "false_certificate":bool(selection is not None and true[selection]<=threshold),
            "mean_cs_width":float(min(2*B,4*B*radii(max(1,count-E),delta,R))),
            "method":"PAC-Bayes-kl singleton iid-pairs + anytime union",
            "cyclic":cyclic}

def audit(seeds=40,offset=20261116,cyclic=False):
    rows=[run(offset+s,cyclic=cyclic) for s in range(seeds)]
    return {"seeds":seeds,"cyclic":cyclic,
            "summary":{"certified":sum(r["certified"] for r in rows),
                       "false_certificates":sum(r["false_certificate"] for r in rows),
                       "mean_labels":float(np.mean([r["labels"] for r in rows])),
                       "mean_cs_width":float(np.mean([r["mean_cs_width"] for r in rows]))},
            "rows":rows}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=40)
    p.add_argument("--cyclic",action="store_true");p.add_argument("--out",default="results/pac_bayes.json")
    a=p.parse_args();v=audit(a.seeds,cyclic=a.cyclic)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(v,indent=2));print(json.dumps(v["summary"],indent=2))

"""Known-change-point nonstationary preference certificate by martingale resets.

Within each of J known epochs, each pair has a stationary Bernoulli win
probability, independently sampled conditional on the adaptive past. The
edge-specific Beta(.5,.5) likelihood martingale is restarted at every
public known change point; allocating delta/(J*E) and union-bounding gives
simultaneous coverage over ALL epochs, pairs, sample counts and post-hoc
policies. This does NOT handle unknown change points.
"""
from __future__ import annotations
import numpy as np,json,argparse
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_cs,edge_coefficients,certificate
from theory_and_guarantees import cyclic_witness

def worlds(k=5):
    if k!=5:raise ValueError("witness K=5")
    b=np.ones(k)/k;W=np.zeros((k,k))
    def put(i,j,x):W[i,j]=x;W[j,i]=-x
    put(0,1,.6)
    for j in (2,3,4):put(0,j,.6);put(1,j,-.4)
    put(2,3,.3);put(3,4,.3);put(2,4,-.3)
    assert W[2,3]>0 and W[3,4]>0 and W[4,2]>0
    qplus=b.copy();qplus[0]+=.08;qplus[1]-=.08
    qminus=b.copy();qminus[0]-=.08;qminus[1]+=.08
    return b,np.stack([qplus,qminus]),np.eye(k),[W,-W]

def trial(seed,epochs=3,epoch_budget=1510,delta=.05,threshold=.015):
    b,Q,H,Wset=worlds();k=len(b)
    E=k*(k-1)//2;edges=list(combinations(range(k),2))
    As=[edge_coefficients(b,q,H)[1] for q in Q]
    cases=[]
    for ep in range(epochs):
        W=Wset[ep%2]
        mu=np.array([W[i,j] for i,j in edges])
        streams=[np.random.default_rng(np.random.SeedSequence([seed,ep,e]))
                 for e in range(E)]
        n=np.ones(E,int)
        s=np.array([streams[e].binomial(1,(1+mu[e])/2) for e in range(E)],int)
        lo=np.zeros(E);hi=np.zeros(E);covered=True;chosen=None
        for e in range(E):lo[e],hi[e]=edge_cs(int(s[e]),int(n[e]),delta/(epochs*E))
        covered=bool(np.all(lo<=mu+1e-12) and np.all(hi>=mu-1e-12))
        while True:
            L=np.array([certificate(A,lo,hi) for A in As]);winner=int(np.argmax(L))
            if L[winner]>threshold:chosen=winner;break
            if n.sum()>=epoch_budget:break
            half=(hi-lo)/2;mid=(lo+hi)/2;P=[]
            for A in As:
                worst=int(np.argmin(A@mid-np.abs(A)@half))
                P.append(np.abs(A[worst])*half/(np.sqrt(n)+.2))
            P=np.array(P)
            score=.7*P[winner]+.3*np.max(P,axis=0)
            e=int(np.argmax(score)) if score.max()>0 else int(np.argmin(n))
            s[e]+=streams[e].binomial(1,(1+mu[e])/2);n[e]+=1
            lo[e],hi[e]=edge_cs(int(s[e]),int(n[e]),delta/(epochs*E))
            covered=bool(covered and lo[e]<=mu[e]+1e-12 and hi[e]>=mu[e]-1e-12)
        gains=[min(float((q-b)@W@h) for h in H) for q in Q]
        cases.append({"epoch":ep,"certified":chosen is not None,"selected":chosen,
                      "correct_selected":bool(chosen is not None and gains[chosen]>threshold),
                      "false_certificate":bool(chosen is not None and gains[chosen]<=threshold),
                      "path_covered":covered,"labels":int(n.sum()),
                      "mean_width":float(np.mean(hi-lo))})
    return {"seed":seed,"epoch_rows":cases,
            "path_covered_all_epochs":all(x["path_covered"] for x in cases)}

def audit(seeds=40,epochs=3,epoch_budget=1510):
    r=[trial(20261116+i,epochs,epoch_budget) for i in range(seeds)]
    summaries={}
    for ep in range(epochs):
        xs=[v["epoch_rows"][ep] for v in r]
        summaries[str(ep)]={"certified":sum(v["certified"] for v in xs),
                            "false_certificates":sum(v["false_certificate"] for v in xs),
                            "mean_labels":float(np.mean([v["labels"] for v in xs])),
                            "mean_terminal_width":float(np.mean([v["mean_width"] for v in xs]))}
    return {"seeds":seeds,"epochs":epochs,"budget_per_epoch":epoch_budget,
            "covered_all_epochs":sum(x["path_covered_all_epochs"] for x in r),
            "by_epoch":summaries,"rows":r}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=40)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--out",default="results/change_point.json")
    a=p.parse_args();v=audit(a.seeds,a.epochs)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(v,indent=2))
    print(json.dumps({k:z for k,z in v.items() if k!="rows"},indent=2))

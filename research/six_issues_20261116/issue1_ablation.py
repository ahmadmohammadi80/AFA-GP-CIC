"""Issue 1: identical-clock, identically-observed joint policy-query ablations."""
import json,argparse,sys
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_cs,edge_coefficients,certificate

def trial(seed,mode="greedy",budget=1510,delta=.05,threshold=.015,null=False,cyclic=False):
    if mode not in ("greedy","joint_active","joint_mix","joint_gap"):raise ValueError("mode")
    k=5;b=np.ones(k)/k;qs=[]
    for a,c in [(0,1),(2,3)]:
        q=b.copy();q[a]+=.08;q[c]-=.08;qs.append(q)
    H=np.eye(k);As=[edge_coefficients(b,q,H)[1] for q in qs]
    pot=np.array([.45,-.35,.25,-.15,-.20]);W=pot[:,None]-pot[None,:]
    for i,j in [(0,2),(2,3),(3,0)]:W[i,j]+=.04;W[j,i]-=.04
    if cyclic:
        W[2,4]=-.18;W[4,2]=.18
        assert W[2,3]>0 and W[3,4]>0 and W[4,2]>0
    if null:W*=0
    es=list(combinations(range(k),2));E=len(es)
    mu=np.array([W[i,j] for i,j in es])
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(E)]
    n=np.ones(E,dtype=int);s=np.array([streams[e].binomial(1,(1+mu[e])/2) for e in range(E)],dtype=int)
    lo=np.zeros(E);hi=np.zeros(E)
    for e in range(E):lo[e],hi[e]=edge_cs(int(s[e]),int(n[e]),delta/E)
    gains=[min(float((q-b)@W@h) for h in H) for q in qs]
    winner=None
    for step in range(budget-E+1):
        lower=np.array([certificate(A,lo,hi) for A in As]);top=int(np.argmax(lower))
        if lower[top]>threshold:winner=top;break
        if n.sum()>=budget:break
        mid=(lo+hi)/2;half=(hi-lo)/2;priority=[]
        for A in As:
            worst=int(np.argmin(A@mid-np.abs(A)@half))
            priority.append(np.abs(A[worst])*half/(np.sqrt(n)+.2))
        P=np.array(priority)
        if mode=="greedy":scores=P[top]
        elif mode=="joint_active":
            plausible=[i for i in range(len(qs)) if lower[i]+2*float(np.max(np.abs(As[i])@half))>=lower[top]]
            scores=np.max(P[plausible],axis=0)
        elif mode=="joint_mix":scores=.7*P[top]+.3*np.max(P,axis=0)
        else:
            d=np.maximum(0,threshold-lower);d/=d.sum()+1e-12
            scores=d@P
        e=int(np.argmax(scores)) if scores.max()>0 else int(np.argmin(n))
        s[e]+=streams[e].binomial(1,(1+mu[e])/2);n[e]+=1
        lo[e],hi[e]=edge_cs(int(s[e]),int(n[e]),delta/E)
    return {"seed":seed,"labels":int(n.sum()),"certified":winner is not None,
            "false_certificate":bool(winner is not None and gains[winner]<=threshold),
            "max_lower":float(max(certificate(A,lo,hi) for A in As)),
            "mean_cs_width":float(np.mean(hi-lo)),
            "sign_cycle":bool(W[2,3]>0 and W[3,4]>0 and W[4,2]>0)}

def audit(seeds=40,offset=20261116,budget=1510,cyclic=False):
    modes=("greedy","joint_active","joint_mix","joint_gap");out={}
    for mode in modes:
        out[mode]=[trial(offset+s,mode,budget,cyclic=cyclic) for s in range(seeds)]
    summary={m:{"certified":sum(v["certified"] for v in rows),
                 "false_certificates":sum(v["false_certificate"] for v in rows),
                 "mean_labels":float(np.mean([v["labels"] for v in rows])),
                 "mean_cs_width":float(np.mean([v["mean_cs_width"] for v in rows]))} for m,rows in out.items()}
    # Exact matched-pairs two-sided McNemar (binomial) tests, then Holm FWER.
    tests={}
    for m in ("joint_active","joint_mix","joint_gap"):
        wins=sum(a["certified"] and not b["certified"] for a,b in zip(out[m],out["greedy"]))
        losses=sum(b["certified"] and not a["certified"] for a,b in zip(out[m],out["greedy"]))
        p=float(binomtest(min(wins,losses),wins+losses,p=.5,alternative="two-sided").pvalue) if wins+losses else 1.
        tests[m]={"joint_only":wins,"greedy_only":losses,"exact_two_sided_p":p}
    order=sorted(tests,key=lambda k:tests[k]["exact_two_sided_p"])
    running=0.
    for rank,m in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*tests[m]["exact_two_sided_p"]))
        tests[m]["holm_adjusted_p"]=running
    return {"seeds":seeds,"offset":offset,"budget":budget,"cyclic":cyclic,"summary":summary,
            "paired_mcnemar_holm":tests,"rows":out}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--offset",type=int,default=20261116)
    p.add_argument("--seeds",type=int,default=40);p.add_argument("--out",default="results/issue1.json")
    p.add_argument("--cyclic",action="store_true")
    a=p.parse_args();r=audit(a.seeds,a.offset,cyclic=a.cyclic);f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(r,indent=2));print(json.dumps({"summary":r["summary"],"paired_mcnemar_holm":r["paired_mcnemar_holm"]},indent=2))

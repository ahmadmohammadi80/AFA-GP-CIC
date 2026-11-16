"""Literature selection policies evaluated with a COMMON anytime safe certificate.

RUCB: Zoghi et al., ICML 2014, https://proceedings.mlr.press/v32/zoghi14.html
D-TS: Wu and Liu, NeurIPS 2016, https://papers.nips.cc/paper/6157
The published algorithms optimize dueling regret / Copeland identification, not our
KL-constrained certificate. Their query selectors are reproduced and the terminal
criterion is necessarily ADAPTED to the common certificate, not passed off as the
published guarantee. RUCB assumes a Condorcet winner; this assumption can fail
in our genuinely cyclic stress case.

Self comparisons from the papers yield no pairwise information; we replace them
by a distinct challenger (documented deviation) to keep matched LABEL budgets.
"""
from __future__ import annotations
import sys,json,math,argparse
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_cs,edge_coefficients,certificate
from issue1_ablation import trial as internal_trial

def preference_problem(cyclic=False):
    k=5;b=np.ones(k)/k;Q=[]
    for a,c in [(0,1),(2,3)]:
        q=b.copy();q[a]+=.08;q[c]-=.08;Q.append(q)
    v=np.array([.45,-.35,.25,-.15,-.20])
    W=v[:,None]-v[None,:]
    for i,j in [(0,2),(2,3),(3,0)]:W[i,j]+=.04;W[j,i]-=.04
    if cyclic:
        # 2 beats 3, 3 beats 4, 4 beats 2: a SIGN cycle, not
        # merely a nonzero curl in an otherwise transitive matrix.
        W[2,4]=-.18;W[4,2]=.18
        assert W[2,3]>0 and W[3,4]>0 and W[4,2]>0
    return b,Q,np.eye(k),W

def pair_matrices(wins,counts,k):
    P=np.full((k,k),.5);N=np.zeros((k,k))
    for idx,(i,j) in enumerate(combinations(range(k),2)):
        n=int(counts[idx]);w=int(wins[idx])
        P[i,j]=(w/n if n else .5);P[j,i]=1-P[i,j];N[i,j]=N[j,i]=n
    return P,N

def choose_pair(method,wins,counts,k,t,rng,champion_memory=None):
    if method not in ("rucb","dts"):raise ValueError(method)
    P,N=pair_matrices(wins,counts,k)
    with np.errstate(divide="ignore",invalid="ignore"):
        rad=np.sqrt(.6*np.log(max(t,2))/np.where(N>0,N,1))
    U=np.where(N>0,P+rad,1.);L=np.where(N>0,P-rad,0.)
    np.fill_diagonal(U,.5);np.fill_diagonal(L,.5)
    if method=="rucb":
        C=np.flatnonzero(np.all(U>=.5,axis=1))
        if len(C)==0:C=np.arange(k)
        if champion_memory is not None and champion_memory in C and rng.random()<.5:
            c=int(champion_memory)
        else:c=int(rng.choice(C))
        rivals=np.array([j for j in range(k) if j!=c])
        d=int(rng.choice(rivals[np.isclose(U[rivals,c],np.max(U[rivals,c]))]))
    else:
        copeland_upper=(U>.5).sum(axis=1)
        C=np.flatnonzero(copeland_upper==copeland_upper.max())
        theta=np.full((k,k),.5)
        for ix,(i,j) in enumerate(combinations(range(k),2)):
            v=rng.beta(wins[ix]+1,counts[ix]-wins[ix]+1)
            theta[i,j]=v;theta[j,i]=1-v
        scores=(theta>.5).sum(axis=1)
        maxima=np.flatnonzero(scores[C]==scores[C].max())
        c=int(rng.choice(C[maxima]))
        rivals=np.array([j for j in range(k) if j!=c and L[j,c]<=.5])
        if len(rivals)==0:rivals=np.array([j for j in range(k) if j!=c])
        second={}
        for j in rivals:
            a,b=(j,c) if j<c else (c,j)
            ix=a*(2*k-a-1)//2+(b-a-1)
            v=rng.beta(wins[ix]+1,counts[ix]-wins[ix]+1)
            second[int(j)]=v if j<c else 1-v
        d=int(rng.choice([j for j in rivals if np.isclose(second[int(j)],max(second.values()))]))
    i,j=sorted((c,d))
    return i*(2*k-i-1)//2+(j-i-1),c

def published_trial(seed,method="rucb",budget=1510,delta=.05,threshold=.015,cyclic=False):
    b,Q,H,W=preference_problem(cyclic);k=len(b)
    es=list(combinations(range(k),2));E=len(es)
    means=np.array([W[i,j] for i,j in es])
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(E)]
    rng=np.random.default_rng(np.random.SeedSequence([seed,7851 if method=="rucb" else 7852]))
    counts=np.ones(E,dtype=int)
    wins=np.array([streams[e].binomial(1,(1+m)/2) for e,m in enumerate(means)],dtype=int)
    lo=np.zeros(E);hi=np.zeros(E)
    for e in range(E):lo[e],hi[e]=edge_cs(int(wins[e]),int(counts[e]),delta/E)
    AA=[edge_coefficients(b,q,H)[1] for q in Q]
    gains=[min(float((q-b)@W@h) for h in H) for q in Q]
    pick=None;champ=None
    while True:
        vals=np.array([certificate(A,lo,hi) for A in AA])
        if max(vals)>threshold:pick=int(np.argmax(vals));break
        if counts.sum()>=budget:break
        e,champ=choose_pair(method,wins,counts,k,int(counts.sum()),rng,champ)
        wins[e]+=streams[e].binomial(1,(1+means[e])/2);counts[e]+=1
        lo[e],hi[e]=edge_cs(int(wins[e]),int(counts[e]),delta/E)
    return {"seed":seed,"method":method,"certified":pick is not None,
            "false_certificate":bool(pick is not None and gains[pick]<=threshold),
            "labels":int(counts.sum()),"mean_cs_width":float(np.mean(hi-lo)),
            "sign_cycle":bool(W[2,3]>0 and W[3,4]>0 and W[4,2]>0)}

def compare(seeds=40,offset=20261116,cyclic=False):
    names=("rucb","dts")
    rows={m:[published_trial(offset+s,m,cyclic=cyclic) for s in range(seeds)] for m in names}
    if cyclic:
        # Avoid comparing cyclic family to old noncyclic joint_mix.
        reference=None
    else:
        reference=[internal_trial(offset+s,"joint_mix") for s in range(seeds)]
        rows["joint_mix"]=reference
    summary={m:{"certified":sum(r["certified"] for r in xs),
                "mean_labels":float(np.mean([r["labels"] for r in xs])),
                "mean_width":(float(np.mean([r["mean_cs_width"] for r in xs]))
                              if m!="joint_mix" else None),
                "false_certificates":sum(r["false_certificate"] for r in xs)}
             for m,xs in rows.items()}
    tests={}
    if reference is not None:
        for m in names:
            w=sum(a["certified"] and not b["certified"] for a,b in zip(reference,rows[m]))
            l=sum(b["certified"] and not a["certified"] for a,b in zip(reference,rows[m]))
            tests[m]={"joint_only":w,"baseline_only":l,
                      "p_exact":float(binomtest(min(w,l),w+l).pvalue) if w+l else 1.}
        ordered=sorted(tests,key=lambda m:tests[m]["p_exact"])
        run=0.
        for i,m in enumerate(ordered):
            run=max(run,min(1.,(len(ordered)-i)*tests[m]["p_exact"]))
            tests[m]["p_holm"]=run
    return {"setup":{"seeds":seeds,"offset":offset,"cyclic":cyclic,
                     "comment":"published query selection with adapted common KL certificate"},
            "summary":summary,"paired":tests,"rows":rows}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,default=40)
    ap.add_argument("--offset",type=int,default=20261116);ap.add_argument("--cyclic",action="store_true")
    ap.add_argument("--out",default="results/published_baselines.json")
    a=ap.parse_args();v=compare(a.seeds,a.offset,a.cyclic)
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
    print(json.dumps({"setup":v["setup"],"summary":v["summary"],"paired":v["paired"]},indent=2))

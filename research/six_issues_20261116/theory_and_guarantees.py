"""Formal finite-action anytime simple-regret/sample bounds and cyclic witness.

The sample guarantee belongs to *guarded_joint* (joint mixture scheduling
plus deterministic minimum-rate coverage), not unguarded joint_mix. This
distinction is essential. No global minimax optimality claim.
"""
from __future__ import annotations
import numpy as np, math,argparse,json
from itertools import combinations
from pathlib import Path
from scipy.stats import binomtest
from published_baselines import preference_problem
from issue1_ablation import trial
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_cs,edge_coefficients,certificate

def clipped_intersection(s,n,delta,E):
    """Split delta/2 to beta-mixture, delta/2 to the all-n Hoeffding union."""
    lo,hi=edge_cs(int(s),int(n),delta/(2*E))
    m=2*s/n-1
    r=math.sqrt(2*math.log(2*E*math.pi**2*n*n/(3*delta))/n)
    a=max(-1.,lo,m-r);b=min(1.,hi,m+r)
    if a>b:
        # Can happen only on the complement of at least one simultaneous
        # valid event. Keep the output well defined; no safety claim there.
        return (-1.,1.)
    return a,b

def radius(n,delta,E):
    if n<1:raise ValueError("positive n")
    return math.sqrt(2*math.log(2*E*math.pi**2*n*n/(3*delta))/n)

def theoretical_sample_cap(b,Q,H,threshold,true_gains,delta=.05,explore_period=20):
    E=len(b)*(len(b)-1)//2
    As=[edge_coefficients(b,q,H)[1] for q in Q]
    S=max(float(np.max(np.abs(A).sum(axis=1))) for A in As)
    gap=max(true_gains)-threshold
    if gap<=0:return {"positive_margin":False,"sample_upper":None}
    nstar=10
    while 2*S*radius(nstar,delta,E)>=gap:
        nstar*=2
        if nstar>1e12:return {"positive_margin":True,"sample_upper":None}
    # Every m-th post-initial query is round-robin, so after at most
    # m*E*(nstar-1) extra queries, every edge has >= nstar samples.
    return {"positive_margin":True,"nstar_conservative":nstar,
            "sample_upper":E+explore_period*E*(nstar-1),
            "coefficient_L1_S":S,"margin":gap,
            "status":"valid but conservative; finite Q,H, known stationary oracle"}

def guarded_joint(seed,budget=1510,cyclic=False,explore_period=20,delta=.05,threshold=.015):
    if explore_period<2:raise ValueError("explore_period")
    b,Q,H,W=preference_problem(cyclic);k=len(b)
    edges=list(combinations(range(k),2));E=len(edges)
    As=[edge_coefficients(b,q,H)[1] for q in Q]
    gains=[min(float((q-b)@W@h) for h in H) for q in Q]
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(E)]
    means=np.array([W[i,j] for i,j in edges])
    n=np.ones(E,int);s=np.array([streams[e].binomial(1,(1+m)/2) for e,m in enumerate(means)],int)
    lo=np.zeros(E);hi=np.zeros(E)
    for e in range(E):lo[e],hi[e]=clipped_intersection(int(s[e]),int(n[e]),delta,E)
    selection=None;rr=0
    while True:
        L=np.array([certificate(A,lo,hi) for A in As]);top=int(np.argmax(L))
        if L[top]>threshold:selection=top;break
        if n.sum()>=budget:break
        post=int(n.sum()-E)
        if (post+1)%explore_period==0:
            e=rr%E;rr+=1
        else:
            mid=(lo+hi)/2;half=(hi-lo)/2;P=[]
            for A in As:
                worst=int(np.argmin(A@mid-np.abs(A)@half))
                P.append(np.abs(A[worst])*half/(np.sqrt(n)+.2))
            P=np.array(P)
            score=.7*P[top]+.3*np.max(P,axis=0)
            e=int(np.argmax(score)) if score.max()>0 else int(np.argmin(n))
        s[e]+=streams[e].binomial(1,(1+means[e])/2);n[e]+=1
        lo[e],hi[e]=clipped_intersection(int(s[e]),int(n[e]),delta,E)
    L=np.array([certificate(A,lo,hi) for A in As]);top=int(np.argmax(L))
    true_best=max(gains)
    hat_gain=gains[top]
    S=max(float(np.max(np.abs(A).sum(axis=1))) for A in As)
    bound=2*S*radius(int(np.min(n)),delta,E)
    return {"seed":seed,"labels":int(n.sum()),"certified":selection is not None,
            "false_certificate":bool(selection is not None and gains[selection]<=threshold),
            "simple_regret":float(true_best-hat_gain),"theory_upper_simple_regret":bound,
            "min_edge_samples":int(np.min(n)),
            "within_bound":bool(true_best-hat_gain<=bound+1e-11),
            "margin":float(true_best-threshold)}

def cyclic_witness(delta=.05,epsilon=.08):
    """Two known worlds: joint policy selection succeeds, frozen q cannot."""
    k=5;b=np.ones(k)/k;W=np.zeros((k,k))
    def put(i,j,x):W[i,j]=x;W[j,i]=-x
    put(0,1,.6)
    for j in (2,3,4):
        put(0,j,.6);put(1,j,-.4)
    put(2,3,.3);put(3,4,.3);put(2,4,-.3)
    assert W[2,3]>0 and W[3,4]>0 and W[4,2]>0
    qplus=b.copy();qplus[0]+=epsilon;qplus[1]-=epsilon
    qminus=b.copy();qminus[0]-=epsilon;qminus[1]+=epsilon
    worlds=(W,-W);policies=(qplus,qminus)
    margin=min(min(float((policies[ix]-b)@worlds[ix]@np.eye(k)[j]) for j in range(k)) for ix in (0,1))
    # 0.8 vs 0.2 Bernoulli on edge (0,1); Hoeffding majority.
    n=math.ceil(math.log(1/delta)/(.18))
    success=[0,0];N=40
    for idx in (0,1):
        for seed in range(N):
            rng=np.random.default_rng(np.random.SeedSequence([seed,idx,9901]))
            win=int(rng.binomial(n,.8 if idx==0 else .2))
            chosen=0 if win>n/2 else 1
            success[idx]+=int(chosen==idx)
    return {"n_diagnostic_comparisons":n,"delta":delta,
            "success_world_plus_40":success[0],"success_world_minus_40":success[1],
            "minimum_true_gain_for_correct_policy":margin,
            "frozen_policy_bayes_success_upper":.5,
            "scope":"known pair of cyclic W worlds, precommitted policy comparator only"}

def audit(seeds=40):
    xs=[guarded_joint(20261116+s) for s in range(seeds)]
    b,Q,H,W=preference_problem()
    true=[min(float((q-b)@W@h) for h in H) for q in Q]
    return {"guarded":{"successes":sum(x["certified"] for x in xs),
                       "false_certificates":sum(x["false_certificate"] for x in xs),
                       "violations_observed":sum(not x["within_bound"] for x in xs),
                       "mean_samples":float(np.mean([x["labels"] for x in xs])),
                       "rows":xs},
            "sample_cap":theoretical_sample_cap(b,Q,H,.015,true),
            "cyclic_witness":cyclic_witness()}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,default=40)
    ap.add_argument("--out",default="results/guarantee.json")
    a=ap.parse_args();r=audit(a.seeds)
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!="rows"} if isinstance(v,dict) else v for k,v in r.items()},indent=2))

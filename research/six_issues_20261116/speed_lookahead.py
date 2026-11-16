"""Issue 6: EXACT cached one-step recourse and analytic frontier KL solver.

The old one_step_scan recomputes every opponent x edge sum after each
hypothetical outcome: O(Q*H*E^2*B) work for E pairs and block size B.
Precompute the Q*H*E old edge contributions ONCE, then each possible
outcome changes only queried edge e. This yields
O(Q*H*E + Q*H*E*B) work and O(Q*H*E) workspace.

For q=(1-eta)b+eta e_i the KL simplifies EXACTLY to
q_i log(q_i/b_i)+(1-q_i) log(1-eta). Each bisection iteration is O(1)
rather than O(K); generating all K candidates remains O(K^2).
"""
from __future__ import annotations
import sys,math,time,tracemalloc,json,argparse
from pathlib import Path
import numpy as np
from scipy.stats import betabinom
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"postquery_recourse_20261116"))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"/"cert_voi"))
from anytime_mixture import edge_cs,edge_coefficients,certificate
from policy_query_recourse import one_step_scan as dense_recourse
from afa_cic_joint import frontier_candidates as dense_frontier

def frontier_fast(b,kappa):
    b=np.asarray(b,float)
    if b.ndim!=1 or np.any(b<=0) or abs(b.sum()-1)>1e-10 or kappa<0:
        raise ValueError("valid positive reference and nonnegative kappa")
    k=len(b);out=np.empty((k,k))
    for i,p in enumerate(b):
        if kappa>=-math.log(p):
            eta=1.
        else:
            low=0.;high=1.
            for _ in range(60):
                eta=(low+high)/2
                z=p+eta*(1-p)
                f=z*math.log(z/p)+(1-z)*math.log1p(-eta)
                if f<=kappa:low=eta
                else:high=eta
            eta=low
        q=(1-eta)*b;q[i]+=eta;out[i]=q
    return out

def recourse_fast(b,Q,H,wins,counts,delta=.05,block=50):
    b=np.asarray(b,float);Q=np.asarray(Q,float);H=np.asarray(H,float)
    wins=np.asarray(wins,int);counts=np.asarray(counts,int)
    E=len(b)*(len(b)-1)//2
    if Q.ndim!=2 or H.ndim!=2 or Q.shape[1]!=len(b) or H.shape[1]!=len(b) or len(wins)!=E or len(counts)!=E or np.any(wins<0) or np.any(wins>counts) or np.any(counts<0) or block<1 or not 0<delta<1:
        raise ValueError("inputs")
    A=np.stack([edge_coefficients(b,q,H)[1] for q in Q]) # Q,H,E
    z=[edge_cs(int(w),int(n),delta/E) for w,n in zip(wins,counts)]
    lo=np.array([a for a,_ in z]);hi=np.array([a for _,a in z])
    oldparts=np.where(A>=0,A*lo,A*hi)
    oldrows=oldparts.sum(axis=2) # Q,H
    recourse=[];precommit=[]
    for e in range(E):
        xs=np.arange(block+1)
        probs=betabinom.pmf(xs,block,wins[e]+.5,counts[e]-wins[e]+.5)
        probs/=probs.sum()
        new_intervals=[edge_cs(int(wins[e]+x),int(counts[e]+block),delta/E) for x in xs]
        newlo=np.array([v[0] for v in new_intervals])
        newhi=np.array([v[1] for v in new_intervals])
        ace=A[:,:,e]
        revised=np.where(ace[:,:,None]>=0,
                         ace[:,:,None]*newlo,ace[:,:,None]*newhi)
        rows=oldrows[:,:,None]-oldparts[:,:,e,None]+revised
        vals=rows.min(axis=1)  # Q x possible outcome
        recourse.append(float(probs@np.max(vals,axis=0)))
        precommit.append(float(np.max(vals@probs)))
    recourse=np.array(recourse);precommit=np.array(precommit)
    return {"recourse":recourse,"precommit":precommit,
            "best_joint_edge":int(np.argmax(recourse)),
            "best_precommit_edge":int(np.argmax(precommit)),
            "global_joint":float(max(recourse)),
            "global_precommit":float(max(precommit)),
            "current_policy_certificates":oldrows.min(axis=1)}

def profile(k=5,q_count=3,block=8,repetitions=2):
    rng=np.random.default_rng(20261116+k)
    b=rng.dirichlet(np.ones(k));Q=rng.dirichlet(np.ones(k),size=q_count)
    H=np.eye(k);E=k*(k-1)//2
    n=np.full(E,30,int);wins=rng.binomial(n,.5)
    old=dense_recourse(b,Q,H,wins,n,.05,block)
    new=recourse_fast(b,Q,H,wins,n,.05,block)
    for field in ("recourse","precommit","current_policy_certificates"):
        np.testing.assert_allclose(old[field],new[field],rtol=0,atol=1e-12)
    slowfront=dense_frontier(b,.05)
    fastfront=frontier_fast(b,.05)
    np.testing.assert_allclose(slowfront,fastfront,rtol=0,atol=1e-8)
    timing={}
    for name,func in (("dense_recourse",dense_recourse),("cached_recourse",recourse_fast)):
        tracemalloc.start();begin=time.perf_counter()
        for _ in range(repetitions):func(b,Q,H,wins,n,.05,block)
        elapsed=(time.perf_counter()-begin)/repetitions
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        timing[name]={"sec":elapsed,"peak_python_bytes":peak}
    return {"k":k,"edges":E,"q":q_count,"block":block,
            "speedup":timing["dense_recourse"]["sec"]/max(1e-12,timing["cached_recourse"]["sec"]),
            "timing":timing,
            "max_numeric_discrepancy":float(np.max(np.abs(old["recourse"]-new["recourse"])))}

def audit():
    rows=[profile(k=k,repetitions=(3 if k<=5 else 1)) for k in (3,5,8,10)]
    return {"method":"exact cached single-edge changes in lookahead","rows":rows}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/lookahead_scale.json")
    a=p.parse_args();r=audit()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k=[x["k"] for x in r["rows"]]
    for name in ("dense_recourse","cached_recourse"):
        plt.plot(k,[1000*x["timing"][name]["sec"] for x in r["rows"]],
                 marker="o",label=name)
    plt.yscale("log");plt.xlabel("Number of responses K")
    plt.ylabel("Measured lookahead milliseconds")
    plt.legend();plt.tight_layout()
    plt.savefig(out.with_suffix(".png"),dpi=180)
    plt.close();print(json.dumps(r,indent=2))

"""Exact one-batch joint policy/query recourse on anytime-valid preference certificates.

The mixture posterior is used for QUERY PLANNING ONLY; the final acceptance
test remains frequentist and anytime valid. One-step recourse inequality is
standard value-of-information, not a claim of global algorithmic novelty.
"""
import json,math,sys
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import betabinom

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"anytime_mixture_20261116"))
from anytime_mixture import edge_cs, edge_coefficients, certificate, run_one as strong_greedy


def one_step_scan(b, Q, H, wins, counts, delta=.05, block=50):
    """Exact Beta(.5,.5) predictive recourse vs q precommit for every edge.

    Returns sup_e E[max_q L_after] and sup_e max_q E[L_after], where each
    expectation uses identical posterior and the same valid confidence bound.
    """
    b=np.asarray(b,float);Q=np.asarray(Q,float);H=np.asarray(H,float)
    wins=np.asarray(wins,int);counts=np.asarray(counts,int)
    if Q.ndim!=2 or H.ndim!=2 or Q.shape[1]!=len(b) or H.shape[1]!=len(b):
        raise ValueError("probability dimensions")
    E=len(counts)
    if E!=len(b)*(len(b)-1)//2 or len(wins)!=E or np.any(counts<0) or np.any(wins<0) or np.any(wins>counts) or block<1 or not 0<delta<1:
        raise ValueError("counts, confidence or block")
    AA=[edge_coefficients(b,q,H)[1] for q in Q]
    z=[edge_cs(int(w),int(n),delta/E) for w,n in zip(wins,counts)]
    lo=np.array([x[0] for x in z]);hi=np.array([x[1] for x in z])
    before=np.array([certificate(A,lo,hi) for A in AA])
    recourse=[];precommit=[]
    for e in range(E):
        xs=np.arange(block+1)
        probs=betabinom.pmf(xs,block,wins[e]+.5,counts[e]-wins[e]+.5)
        probs=probs/probs.sum()
        outcomes=[]
        for v in xs:
            l,h=edge_cs(int(wins[e]+v),int(counts[e]+block),delta/E)
            le,he=lo.copy(),hi.copy();le[e]=l;he[e]=h
            outcomes.append([certificate(A,le,he) for A in AA])
        vals=np.array(outcomes)
        recourse.append(float(probs@np.max(vals,axis=1)))
        precommit.append(float(np.max(probs@vals)))
    return {"recourse":np.asarray(recourse),"precommit":np.asarray(precommit),
            "best_joint_edge":int(np.argmax(recourse)),
            "best_precommit_edge":int(np.argmax(precommit)),
            "global_joint":max(recourse),"global_precommit":max(precommit),
            "current_policy_certificates":before}


def replay(seed,mode="joint",budget=1510,block=50,delta=.05,threshold=.015):
    """Paired stationary-stream benchmark, identical q,H,W,b,delta,budget.

    'precommit' can reselect its q in the next round. 'joint' integrates
    prospective q re-selection in its one-block edge score. 'greedy_batch'
    has exactly the same block-size decision schedule as both VOI methods.
    """
    if mode not in ("joint","precommit","greedy_batch") or block<1 or budget<10:
        raise ValueError("mode, block or budget")
    k=5;b=np.ones(k)/k
    q1=b.copy();q1[0]+=.08;q1[1]-=.08
    q2=b.copy();q2[2]+=.08;q2[3]-=.08
    Q=np.array([q1,q2]);H=np.eye(k)
    kl=[float(np.sum(q*np.log(q/b))) for q in Q]
    if max(kl)>.05:raise AssertionError("KL constraint")
    potential=np.array([.45,-.35,.25,-.15,-.20])
    W=potential[:,None]-potential[None,:]
    for i,j in [(0,2),(2,3),(3,0)]:
        W[i,j]+=.04;W[j,i]-=.04
    es=list(combinations(range(k),2))
    true=np.array([W[i,j] for i,j in es])
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(es))]
    counts=np.ones(len(es),int)
    wins=np.array([streams[e].binomial(1,(1+v)/2) for e,v in enumerate(true)],int)
    AA=[edge_coefficients(b,q,H)[1] for q in Q]
    winner=None
    while counts.sum()<=budget:
        intervals=[edge_cs(int(s),int(n),delta/len(es)) for s,n in zip(wins,counts)]
        lo=np.array([z[0] for z in intervals]);hi=np.array([z[1] for z in intervals])
        certs=np.array([certificate(A,lo,hi) for A in AA])
        if max(certs)>threshold:
            winner=int(np.argmax(certs));break
        remaining=budget-int(counts.sum())
        if remaining==0:break
        steps=min(block,remaining)
        if mode=="greedy_batch":
            # Same beta-mixture controller as greedy_online, but forced to
            # select only once per block: same opportunities as the VOI arms.
            A=AA[int(np.argmax(certs))]
            center=(lo+hi)/2
            worst=int(np.argmin(A@center-np.abs(A)@((hi-lo)/2)))
            priority=np.abs(A[worst])*(hi-lo)/(np.sqrt(counts)+.2)
            e=int(np.argmax(priority)) if priority.max()>0 else int(np.argmin(counts))
        else:
            scan=one_step_scan(b,Q,H,wins,counts,delta,steps)
            e=scan["best_joint_edge"] if mode=="joint" else scan["best_precommit_edge"]
        # Generate each Bernoulli observation separately. Unlike binomial(B,p),
        # this ensures exactly identical per-edge RNG prefixes across block sizes.
        wins[e]+=int(streams[e].binomial(1,(1+true[e])/2,size=steps).sum())
        counts[e]+=steps
    true_gains=[float(min((q-b)@W@h for h in H)) for q in Q]
    return {"mode":mode,"labels":int(counts.sum()),"certified":winner is not None,
            "selected":winner,"false_positive":bool(winner is not None and true_gains[winner]<=threshold),
            "max_lower":float(max(certs)),"true_gains":true_gains,"KL":kl}


def experiment(seeds=40,budget=1510,block=50):
    methods=["joint","precommit","greedy_batch","greedy_online"]
    rows=[];summaries={}
    for s in range(seeds):
        seed=20261116+s
        row={m:replay(seed,m,budget,block) for m in methods[:3]}
        baseline=strong_greedy(seed=seed,max_labels=budget,k=5,delta=.05,threshold=.015,mode="mixture")
        row["greedy_online"]={"labels":baseline["labels"],"certified":baseline["certified"],
                              "false_positive":baseline["certified"] and baseline["true_gain"]<=.015}
        rows.append(row)
    for m in methods:
        summaries[m]={"certificates":sum(r[m]["certified"] for r in rows),
                      "false_certificates":sum(r[m]["false_positive"] for r in rows),
                      "mean_labels":float(np.mean([r[m]["labels"] for r in rows]))}
    return {"scope":"fixed candidates; one-block policy-recourse, not novelty or sample-optimality proof",
            "seeds":seeds,"budget":budget,"block":block,
            "methods":summaries,"rows":rows}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=40)
    p.add_argument("--budget",type=int,default=1510)
    p.add_argument("--block",type=int,default=50)
    p.add_argument("--out",default="results/recourse_audit.json")
    args=p.parse_args()
    result=experiment(args.seeds,args.budget,args.block)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))

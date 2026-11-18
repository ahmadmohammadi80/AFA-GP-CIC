"""Exact static-parameter two-query planner integrated with the existing beta-mixture
anytime policy-improvement certificate used by issue1_ablation.py.

No new policy proposal, no new confidence sequence: same fixed policies, finite
opponents, comparison streams and confidence bounds as the original benchmark.
The worst-case objective is over one fixed pairwise-preference kernel. It is not
an uncertainty-adaptive (rectangular) adversary reselecting p after each label.
Run: python integrated_two_step_certificate_20261118.py --seeds 40 --budget 1510
"""
import argparse
from itertools import combinations, product
from pathlib import Path
import sys,time,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'anytime_mixture_20261116'))
from anytime_mixture import edge_cs, certificate, edge_coefficients


def intervals(s,n,delta):
    E=len(n)
    return np.array([edge_cs(int(s[e]),int(n[e]),delta/E) for e in range(E)])


def exact_static_min(box,e,a,b,flags):
    """Exact min over one fixed kernel p. Others enter multi-affinely, root <=quadratically."""
    box=np.asarray(box,float)
    if not np.all((0<=box[:,0]) & (box[:,0]<=box[:,1]) & (box[:,1]<=1)):
        raise ValueError('invalid probability box')
    others=sorted({a,b}-{e})
    best=float('inf')
    for ends in product((0,1),repeat=len(others)):
        p=box[:,0].copy()
        for j,side in zip(others,ends):p[j]=box[j,side]
        def f(x):
            p[e]=x
            return (1-x)*((1-p[a])*flags[0]+p[a]*flags[1])+x*((1-p[b])*flags[2]+p[b]*flags[3])
        c,v,h=f(0),f(1),f(.5)
        A=2*(v+c-2*h);B=v-c-A
        l,u=box[e]
        best=min(best,f(l),f(u))
        if A>1e-14:
            stationary=-B/(2*A)
            if l<stationary<u:best=min(best,f(stationary))
    return float(max(0,min(1,best)))


def leaf_certifier(s,n,As,delta,threshold):
    """The original beta-mixture CS and identical finite-policy / opponent certificate."""
    E=len(n)
    cache={}
    def terminal(history):
        key=tuple(history)
        if key in cache:return cache[key]
        ss=s.copy();nn=n.copy()
        for e,y in key:
            ss[e]+=y;nn[e]+=1
            ci=intervals(ss,nn,delta)
            lo,hi=ci[:,0],ci[:,1]
            if any(certificate(A,lo,hi)>threshold for A in As):
                cache[key]=1;return 1
        cache[key]=0;return 0
    return terminal


def plan_two(s,n,As,delta=.05,threshold=.015):
    """All m^3 adaptive deterministic trees under a common fixed-kernel box."""
    ci=intervals(s,n,delta)
    box=(ci+1)/2
    term=leaf_certifier(s,n,As,delta,threshold)
    if any(certificate(A,ci[:,0],ci[:,1])>threshold for A in As):
        return {'already_certified':True, 'optimal':None}
    m=len(n)
    flags={}
    for e in range(m):
        for j in range(m):
            for y in (0,1):
                flags[e,j,y]=[term(((e,y),(j,0))),term(((e,y),(j,1)))]
    best=(-1,None);roots=[(-1,None) for _ in range(m)]
    for e in range(m):
        for a in range(m):
            for b in range(m):
                f=[*flags[e,a,0],*flags[e,b,1]]
                v=exact_static_min(box,e,a,b,f)
                item={'e':e,'a':a,'b':b,'robust_probability':v}
                if v>best[0]+1e-14:best=(v,item)
                if v>roots[e][0]+1e-14:roots[e]=(v,item)
    return {'already_certified':False,'optimal':best[1],
            'best_for_root':[v[1] for v in roots], 'initial_lower':max(certificate(A,ci[:,0],ci[:,1]) for A in As)}


def greedy_edge(s,n,As,delta=.05):
    """Exactly the greedy selection score of issue1_ablation.py."""
    ci=intervals(s,n,delta);lo=ci[:,0];hi=ci[:,1]
    lower=np.array([certificate(A,lo,hi) for A in As]);top=int(np.argmax(lower))
    mid=(lo+hi)/2;half=(hi-lo)/2;A=As[top]
    worst=int(np.argmin(A@mid-np.abs(A)@half))
    scores=np.abs(A[worst])*half/(np.sqrt(n)+.2)
    return int(np.argmax(scores)) if scores.max()>0 else int(np.argmin(n))


def canonical_problem(seed,cyclic=False):
    """Same 5 responses, candidate policies, opponents, W, prefix oracle as original."""
    k=5;ref=np.ones(k)/k;qs=[]
    for i,j in ((0,1),(2,3)):
        q=ref.copy();q[i]+=.08;q[j]-=.08;qs.append(q)
    assert all(sum(q*np.log(q/ref))<.05 for q in qs)
    H=np.eye(k);As=[edge_coefficients(ref,q,H)[1] for q in qs]
    pot=np.array([.45,-.35,.25,-.15,-.20]);W=pot[:,None]-pot[None,:]
    for i,j in ((0,2),(2,3),(3,0)):W[i,j]+=.04;W[j,i]-=.04
    if cyclic:W[2,4]=-.18;W[4,2]=.18
    es=list(combinations(range(k),2));mu=np.array([W[i,j] for i,j in es])
    streams=[np.random.default_rng(np.random.SeedSequence([seed,e])) for e in range(len(es))]
    n=np.ones(len(es),dtype=int)
    s=np.array([streams[e].binomial(1,(1+mu[e])/2) for e in range(len(es))],dtype=int)
    return s,n,streams,mu,As,W,qs,ref,H


def boundary_audit(seed=20261116,budget=1200,delta=.05,threshold=.015,cyclic=False):
    """Greedy prefix, then evaluate exactly both options on SAME state and 2-query budget.
    Planning once near budget end tests integration, not full receding-horizon efficacy.
    """
    if budget<12:raise ValueError('budget must provide >=2 extra labels')
    s,n,streams,mu,As,W,qs,ref,H=canonical_problem(seed,cyclic)
    # The state is selected by a predeclared, history-adapted gate: stop
    # before the next query once the gap is <= .0005 below the threshold.
    # It is a diagnostic near-boundary audit, not a full online benchmark.
    while n.sum()<budget-2:
        ci=intervals(s,n,delta)
        margin=threshold-max(certificate(A,ci[:,0],ci[:,1]) for A in As)
        if margin<0 or margin<=.0005:break
        e=greedy_edge(s,n,As,delta)
        s[e]+=streams[e].binomial(1,(1+mu[e])/2);n[e]+=1
    t0=time.perf_counter();plan=plan_two(s,n,As,delta,threshold)
    seconds=time.perf_counter()-t0
    if plan['already_certified']:
        return {'seed':seed,'already_certified':True,'labels':int(n.sum()),'planning_seconds':round(seconds,5), 's':s.tolist(),'n':n.tolist()}
    g=greedy_edge(s,n,As,delta)
    gv=plan['best_for_root'][g]['robust_probability'];ov=plan['optimal']['robust_probability']
    return {'seed':seed,'already_certified':False,'labels':int(n.sum()),'s':s.tolist(),'n':n.tolist(),
            'optimal':plan['optimal'],'greedy_root':g,'greedy_with_optimal_recourse_value':gv,
            'strict_advantage':bool(ov>gv+1e-12),'planning_seconds':round(seconds,5)}


def audit(seeds=2,seed_offset=20261116,budget=1200,cyclic=False):
    rows=[boundary_audit(seed_offset+i,budget,cyclic=cyclic) for i in range(seeds)]
    return {'protocol':{'action_count':5,'candidate_count':2,'opponents':5,
        'confidence':'beta-mixture anytime','KL_cap':.05,'delta':.05,'boundary_gate':.0005,
        'threshold':.015,'total_label_budget':budget,'matching':'shared Bernoulli prefixes',
        'objective':'exact 2-query worst-static-kernel certificate crossing; greedy prefix',
        'status':'greedy-generated near-threshold diagnostic snapshots, not full online two-step outcome benchmark'},
        'rows':rows,'strict_advantages':sum(r.get('strict_advantage',False) for r in rows),
        'positive_planner_values':sum((not r['already_certified']) and r['optimal']['robust_probability']>1e-12 for r in rows)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--budget',type=int,default=1510)
    parser.add_argument('--offset',type=int,default=20261116)
    parser.add_argument('--cyclic',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(seeds=args.seeds,seed_offset=args.offset,budget=args.budget,cyclic=args.cyclic),ensure_ascii=False,indent=2))

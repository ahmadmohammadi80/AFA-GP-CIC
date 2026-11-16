"""AFA certificate-aware edge allocation: finite stationary preference games.

This is a scoped candidate contribution, NOT globally established novelty.
Independent pilot designs the counts; fresh confirmation labels validate safety.
"""
import argparse,json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.optimize import minimize

def coefficients(b,q,H):
    b=np.asarray(b,float);q=np.asarray(q,float);H=np.asarray(H,float)
    if H.ndim!=2 or H.shape[1]!=len(b) or not np.isclose(sum(b),1) or not np.isclose(sum(q),1):
        raise ValueError("probability dimensions")
    d=q-b
    E=list(combinations(range(len(b)),2))
    A=np.asarray([[d[i]*h[j]-d[j]*h[i] for i,j in E] for h in H])
    keep=np.max(abs(A),axis=0)>1e-14
    return [e for e,ok in zip(E,keep) if ok], A[:,keep]

def one_opponent_exact(w,B):
    w=np.asarray(w,float)
    if B<=0 or np.any(w<0):raise ValueError("width")
    z=w**(2/3);S=z.sum()
    return (S/B)**2*z if S else np.zeros_like(w)

def dual_bound(A,B,lam):
    A=np.abs(np.asarray(A,float));B=np.asarray(B,float);lam=np.asarray(lam,float)
    if np.any(lam<0) or A.shape!=(len(B),A.shape[1]) or len(lam)!=len(B):
        raise ValueError("dual dimensions")
    return float(3*2**(-2/3)*np.sum((lam@A)**(2/3))-lam@B)

def design(A,B):
    """Convex log-count optimization; reports feasible primal and dual lower."""
    W=np.abs(np.asarray(A,float));B=np.asarray(B,float)
    if W.ndim!=2 or W.shape[0]!=len(B) or W.shape[1]==0 or np.any(B<=0):
        raise ValueError("design")
    if np.any(np.max(W,axis=0)<=0):raise ValueError("inactive edges")
    E=W.shape[1]
    uniform=max((sum(row)/bi)**2 for row,bi in zip(W,B))
    # Scale by the fully informed uniform feasible count to condition the NLP.
    scale=max(float(uniform),1e-10)
    x0=np.zeros(E)
    normalized=W/(B[:,None]*np.sqrt(scale))
    cons=[{'type':'ineq','fun':lambda x,w=w:float(1-np.sum(w*np.exp(-.5*x))),
           'jac':lambda x,w=w:.5*w*np.exp(-.5*x)}
          for w in normalized]
    opt=minimize(lambda x:float(np.exp(x).sum()),x0,
         jac=lambda x:np.exp(x),bounds=[(-18,18)]*E,constraints=cons,
         method='SLSQP',options={'maxiter':1500,'ftol':1e-11})
    n=scale*np.exp(opt.x) if np.all(np.isfinite(opt.x)) else np.full(E,uniform*1.01)
    if np.max(W@np.power(n,-.5)-B)>1e-7:
        n=np.full(E,uniform*1.01)
    if n.sum()>E*uniform*1.01:n=np.full(E,uniform*1.01)
    N=np.maximum(1,np.ceil(n).astype(int))
    assert np.max(W@N.astype(float)**(-.5)-B)<1e-7
    # Exact primal LB by retaining only one of the simultaneous constraints.
    L=max((np.power(w,2/3).sum()**3)/(t**2) for w,t in zip(W,B))
    total=int(N.sum())
    same=np.full(E,total//E,dtype=int);same[:total%E]+=1
    return {'optimal_counts':N,'uniform_counts':same,'continuous_cost':float(n.sum()),
            'global_lower':float(L),'numerical_gap_upper':float(n.sum()-L),
            'uniform_required_for_same_width':int(E*np.ceil(uniform)),
            'solver_success':bool(opt.success)}

def certificate(A,observations,N,delta):
    N=np.asarray(N,int);A=np.asarray(A,float)
    if np.any(N<=0):raise ValueError("confirmation must be positive")
    c=np.sqrt(2*np.log(2*len(N)/delta))
    return float(np.min(A@observations-c*np.abs(A)@(1/np.sqrt(N))))

def run(trials=40,pilot=100,k=10,delta=.05,threshold=.02,seed=20261116):
    if k<5 or k>13 or pilot<1 or trials<1:raise ValueError("scenario")
    b=np.ones(k)/k;d=np.zeros(k);d[0]=.065;d[1]=-.065;d[2]=.015;d[3]=-.015;q=b+d
    kappa=.10
    policy_kl=float(np.sum(q*np.log(q/b)))
    if policy_kl>kappa:raise AssertionError("candidate policy violates KL budget")
    H=np.eye(k)[:6]
    v=np.linspace(.12,-.10,k);v[0]=.40;v[1]=-.40
    W=v[:,None]-v[None,:]
    for i,j in ((2,3),(3,4),(4,2)):
        W[i,j]+=.12;W[j,i]-=.12
    assert abs(W).max()<=1.
    E,A=coefficients(b,q,H)
    mu=np.array([W[i,j] for i,j in E])
    c=np.sqrt(2*np.log(2*len(E)/delta))
    rows=[]
    for t in range(trials):
        rng=np.random.default_rng(seed+t)
        pilot_m=np.array([2*rng.binomial(pilot,(1+W[i,j])/2)/pilot-1 for i,j in E])
        gap=A@pilot_m-threshold
        if np.min(gap)<=.003:
            rows.append({'seed':t,'planned':False,'pilot_min_gap':float(min(gap))})
            continue
        B=gap/(2*c)
        alloc=design(A,B)
        N=alloc['optimal_counts'];Un=alloc['uniform_counts']
        obs=np.array([2*rng.binomial(n,(1+W[i,j])/2)/n-1 for (i,j),n in zip(E,N)])
        obs_uni=np.array([2*rng.binomial(n,(1+W[i,j])/2)/n-1 for (i,j),n in zip(E,Un)])
        l=certificate(A,obs,N,delta);u=certificate(A,obs_uni,Un,delta)
        rows.append({'seed':t,'planned':True,'pilot':int(pilot*len(E)),'confirm_budget':int(sum(N)),
           'our_lower':l,'uniform_lower':u,'our_certified':bool(l>threshold),
           'uniform_certified':bool(u>threshold),
           'max_width_ours':float(c*np.max(np.abs(A)@N**(-.5))),
           'max_width_uniform':float(c*np.max(np.abs(A)@Un**(-.5))),
           'oracle_design_lower_bound':alloc['global_lower'],
           'uniform_required_for_same_width':alloc['uniform_required_for_same_width'],
           'optimizer_success':alloc['solver_success']})
    valid=[r for r in rows if r['planned']]
    return {'model':'fixed KL candidate and opponent set, independent pilot and confirmation',
       'k':k,'opponent_count':len(H),'active_edges':len(E),'trial_count':trials,
       'kappa':kappa,'policy_kl':policy_kl,
       'true_margins':(A@mu).tolist(),'threshold':threshold,'delta':delta,
       'planned_trials':len(valid),
       'our_certificates':sum(r['our_certified'] for r in valid),
       'uniform_active_edge_certificates':sum(r['uniform_certified'] for r in valid),
       'average_our_width':float(np.mean([r['max_width_ours'] for r in valid])) if valid else None,
       'mean_confirm_budget':float(np.mean([r['confirm_budget'] for r in valid])) if valid else None,
       'mean_uniform_required_budget_for_same_target':float(np.mean([r['uniform_required_for_same_width'] for r in valid])) if valid else None,
       'optimized_below_uniform_target_count':sum(r['confirm_budget']<r['uniform_required_for_same_width'] for r in valid),
       'optimizer_success_count':sum(r['optimizer_success'] for r in valid),
       'average_uniform_width':float(np.mean([r['max_width_uniform'] for r in valid])) if valid else None,
       'rows':rows}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/afa_edge_allocation.json')
    p.add_argument('--trials',type=int,default=40)
    p.add_argument('--pilot',type=int,default=100)
    args=p.parse_args();out=run(trials=args.trials,pilot=args.pilot)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

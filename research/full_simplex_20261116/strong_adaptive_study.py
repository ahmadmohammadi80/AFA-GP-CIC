"""Executable finite-action, preference-cyclic safety-certification study.

Deliberately compares against an oracle pair-transfer algorithm with exact W,
and an unrestricted full-information oracle. This proves a policy-class
separation only. No claim of separation from arbitrary unrestricted adaptivity.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from math import log
import numpy as np
from scipy.optimize import minimize


def kl(q,b):
    q=np.asarray(q,float); b=np.asarray(b,float)
    idx=q>0
    return float(np.sum(q[idx]*np.log(q[idx]/b[idx])))


def cycle_instance(k=24,a=.3,cycle=.25):
    assert k%2==0 and k>=6 and 0<a<=.49
    b=np.full(k,1./k)
    target=np.r_[np.full(k//2,a),np.full(k//2,-a)]
    W=target[:,None]-target[None,:]
    # rock-paper-scissors cycle within high-scoring group, zero Borda effect
    for i,j in [(0,1),(1,2),(2,0)]:
        W[i,j]=cycle; W[j,i]=-cycle
    assert np.allclose(W,-W.T)
    assert np.allclose(W@b,target)
    assert W[0,1]>0 and W[1,2]>0 and W[2,0]>0
    assert np.max(np.abs(W))<=1
    return b,W,target


def bound(n,k,delta=.05):
    if n==0: return 1.
    return min(1., float(np.sqrt(2*np.log(np.pi**2*k*n*n/(3*delta))/n)))


def sample_score(rng,W,b,i,n):
    J=rng.choice(len(b),size=n,p=b)
    p=(W[i,J]+1.)/2.
    out=2*(rng.random(n)<p).astype(float)-1.
    out[J==i]=0.
    return out


def sample_confidence(rng,W,b,n,delta=.05):
    k=len(b)
    sums=np.array([sample_score(rng,W,b,i,n).sum() for i in range(k)])
    means=sums/n
    rad=bound(n,k,delta)
    lo=np.maximum(-1.,means-rad)
    hi=np.minimum(1.,means+rad)
    return means,lo,hi


def robust_gain(q,b,lo,hi):
    d=np.asarray(q)-np.asarray(b)
    return float(np.where(d>=0,d*lo,d*hi).sum())


def oracle_kl_policy(s,b,kappa):
    if kappa<=0:return b.copy()
    if np.max(s)-np.min(s)<1e-14:return b.copy()
    lo,hi=0.,1.
    def qfrom(x):
        t=x*(s-np.max(s))
        e=np.exp(np.log(b)+t); return e/e.sum()
    while kl(qfrom(hi),b)<kappa and hi<1.e7:hi*=2
    for _ in range(60):
        mid=(lo+hi)/2
        if kl(qfrom(mid),b)<=kappa:lo=mid
        else:hi=mid
    return qfrom(lo)


def pair_transfer(b,i,j,kappa):
    if i==j:return 0.
    lo,hi=0.,float(b[j])
    def q(t):
        res=b.copy();res[i]+=t;res[j]-=t;return res
    if kl(q(hi),b)<=kappa:return hi
    for _ in range(55):
        mid=(lo+hi)/2
        if kl(q(mid),b)<=kappa:lo=mid
        else:hi=mid
    return lo


def strongest_pair_oracle(b,s,kappa):
    # EXACT truth-aware optimization over all one-transfer policies;
    # any adaptive strategy limited to one-transfer cannot do better.
    k=len(b)
    best=0.;direction=None
    for i in range(k):
        for j in range(k):
            if i==j or s[i]<=s[j]:continue
            t=pair_transfer(b,i,j,kappa)
            score=t*(s[i]-s[j])
            if score>best:best=score;direction=(i,j,t)
    return best,direction


def robust_full_policy(b,lower,upper,kappa,init=None):
    """Numerical convex optimization of max lower gain on KL ball.

    The mathematical problem is convex (minimization of a convex nonsmooth
    negative certificate over a KL convex sublevel set). Numerical solutions
    are explicitly checked for KL feasibility; no unverified optimality claim.
    """
    k=len(b)
    x0=oracle_kl_policy((lower+upper)/2,b,kappa*.90) if init is None else init
    if kl(x0,b)>kappa:x0=b.copy()
    def fun(q):return -robust_gain(q,b,lower,upper)
    constraints=[{'type':'eq','fun':lambda q:float(np.sum(q)-1.),'jac':lambda q:np.ones_like(q)},
                 {'type':'ineq','fun':lambda q:float(kappa-kl(q,b)),
                  'jac':lambda q:-np.log(np.maximum(q,1e-12)/b)-1.}]
    sols=[]
    for start in (x0,b.copy()):
        res=minimize(fun,start,method='SLSQP',bounds=[(1e-10,1.)]*k,
            constraints=constraints, options={'ftol':1e-10,'maxiter':800})
        if abs(sum(res.x)-1.)<1e-6 and kl(res.x,b)<=kappa+1e-8:
            sols.append(res.x)
    if not sols:raise RuntimeError('no feasible solution from SLSQP')
    best=min(sols,key=fun)
    if kl(best,b)>kappa:
        # Tiny numerical discrepancy conservatively interpolated with b.
        l,r=0.,1.
        for _ in range(50):
            mid=(l+r)/2
            if kl(b+mid*(best-b),b)<=kappa:l=mid
            else:r=mid
        best=b+l*(best-b)
    assert robust_gain(best,b,lower,upper)<=1.0000001
    return best


def run(k=24,n=4000,seeds=30,kappa=.04,gamma=.04,delta=.05):
    b,W,s=cycle_instance(k)
    oracle_q=oracle_kl_policy(s,b,kappa)
    oracle_gain=float(oracle_q@s)
    pair_gain,pair=strongest_pair_oracle(b,s,kappa)
    out=[]
    for seed in range(seeds):
        rng=np.random.default_rng(8102026+1000*k+seed)
        means,lower,upper=sample_confidence(rng,W,b,n,delta)
        q=robust_full_policy(b,lower,upper,kappa)
        cert=robust_gain(q,b,lower,upper)
        true_gain=float(q@s)
        covered=bool(np.all(lower<=s+1e-10) and np.all(s<=upper+1e-10))
        out.append({'seed':seed,'labels':k*n,'certified':bool(cert>gamma),
                    'gain_lower':cert,'true_gain':true_gain,'kl':kl(q,b),'joint_covered':covered})
    return {'model':'cyclic finite action preferences with nonzero RPS 3-cycle',
        'k':k,'n_per_action':n,'number_trials':seeds,'total_queries_per_trial':k*n,
        'kappa':kappa,'required_gain':gamma,'delta':delta,
        'oracle_unrestricted_gain':oracle_gain,
        'oracle_one_transfer_gain':pair_gain,'oracle_one_transfer_direction':pair,
        'proof_upper_bound_pair':.6*min(1./k,np.sqrt(kappa/k)),
        'constructive_full_lower_bound':.3*np.sqrt(kappa),
        'certified_count':int(sum(r['certified'] for r in out)),
        'coverage_count':int(sum(r['joint_covered'] for r in out)),
        'average_certified_gain_lower':float(np.mean([r['gain_lower'] for r in out])),
        'average_true_gain':float(np.mean([r['true_gain'] for r in out])),
        'max_kl':float(max(r['kl'] for r in out)),'trials':out}


def test():
    for k in (6,12,24,48):
        b,W,s=cycle_instance(k)
        assert np.allclose(W@b,s)
        full=oracle_kl_policy(s,b,.04)
        p,_=strongest_pair_oracle(b,s,.04)
        assert kl(full,b)<=.04+1e-9
        assert float(full@s)>p
        assert p<=.6*min(1./k,np.sqrt(.04/k))+1e-8
        assert float(full@s)>=.3*np.sqrt(.04)
    print('Mathematical construction tests passed.')


if __name__=='__main__':
    ar=argparse.ArgumentParser()
    ar.add_argument('--n',type=int,default=4000)
    ar.add_argument('--seeds',type=int,default=30)
    ar.add_argument('--ks',type=int,nargs='+',default=[6,12,24,48])
    ar.add_argument('--out',default='cyclic_study.json')
    args=ar.parse_args();test()
    results=[run(k=k,n=args.n,seeds=args.seeds) for k in args.ks]
    with open(args.out,'w') as f:json.dump(results,f,indent=2)
    for r in results:print({a:b for a,b in r.items() if a!='trials'})

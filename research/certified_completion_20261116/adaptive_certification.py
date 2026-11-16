"""Anytime-safe adaptive certificate controller for finite-action preferences.

Each query asks whether response i is preferred to independent J~b.
The selection of i, output policy q, and stopping time can depend on previous
observations; the simultaneous Hoeffding score intervals remain valid assuming
stationary, conditionally independent labels and known fixed b.

The query heuristic is NOT proved optimal in label complexity.
"""
from dataclasses import dataclass
import numpy as np
from certified_solver import solve,kl
from strong_adaptive_study import oracle_kl_policy, sample_score

@dataclass
class Run:
    certified:bool
    policy:np.ndarray
    lower_bound:float
    dual_upper:float
    optimality_gap:float
    labels:int
    counts:list
    true_gain:float
    kl:float
    query_steps:int


class Controller:
    def __init__(self,b,kappa=.04,gamma=.02,delta=.05,block=25,explore_every=5):
        b=np.asarray(b,float)
        if b.ndim!=1 or len(b)<2 or np.any(b<=0) or not np.isclose(b.sum(),1):
            raise ValueError('invalid full-support b')
        if not 0<delta<1 or kappa<0 or gamma<0 or block<1 or explore_every<1:
            raise ValueError('invalid controls')
        self.b=b.copy();self.k=len(b);self.kappa=kappa;self.gamma=gamma
        self.delta=delta;self.block=int(block);self.explore_every=int(explore_every)
        self.counts=np.zeros(self.k,int);self.sums=np.zeros(self.k,float)
        self.steps=0

    def bounds(self):
        n=self.counts
        center=np.divide(self.sums,n,out=np.zeros(self.k),where=n>0)
        radius=np.full(self.k,2.)
        m=n>0
        radius[m]=np.minimum(2.,np.sqrt(2*np.log(np.pi**2*self.k*n[m].astype(float)**2/(3*self.delta))/n[m]))
        return center,np.maximum(-1.,center-radius),np.minimum(1.,center+radius),radius

    def certificate(self):
        _,lo,hi,_=self.bounds()
        return solve(self.b,lo,hi,self.kappa)

    def choose_next(self):
        # Forced exploration prevents any arm from being forgotten forever.
        if self.steps%self.explore_every==0 or not np.all(self.counts):
            return int(np.argmin(self.counts))
        center,lo,hi,r=self.bounds()
        feasible_q=solve(self.b,lo,hi,self.kappa).q
        optimistic_q=oracle_kl_policy(center,self.b,self.kappa)
        # Exploit the current certificate and plausible favorable policy.
        coeff=np.abs(feasible_q-self.b)+np.abs(optimistic_q-self.b)
        n=self.counts.astype(float)
        after=n+self.block
        after_rad=np.minimum(2.,np.sqrt(2*np.log(np.pi**2*self.k*after**2/(3*self.delta))/after))
        reduction=np.maximum(0.,r-after_rad)
        target=coeff*reduction
        if target.max()<=1e-14:return int(np.argmin(self.counts))
        return int(np.argmax(target))

    def observe(self,i,values):
        values=np.asarray(values,float)
        if not 0<=i<self.k or values.ndim!=1 or len(values)==0 or np.any(~np.isfinite(values)) or np.any(np.abs(values)>1):
            raise ValueError('invalid preference observations')
        self.counts[i]+=len(values);self.sums[i]+=float(values.sum());self.steps+=1

    def fit(self,W,seed=0,budget=6000):
        W=np.asarray(W,float)
        if W.shape!=(self.k,self.k) or not np.allclose(W,-W.T) or np.max(abs(W))>1+1e-12:
            raise ValueError('W must be a skew-symmetric preference matrix with entries [-1,1]')
        if budget<0:raise ValueError('negative budget')
        rng=np.random.default_rng(seed)
        while self.counts.sum()<budget:
            result=self.certificate()
            if result.primal>self.gamma:break
            i=self.choose_next();n=min(self.block,budget-int(self.counts.sum()))
            obs=sample_score(rng,W,self.b,i,n)
            self.observe(i,obs)
        result=self.certificate()
        q=result.q if result.primal>self.gamma else self.b.copy()
        v=float(q@W@self.b)
        return Run(certified=bool(result.primal>self.gamma),policy=q,lower_bound=result.primal,
                   dual_upper=result.dual_upper,optimality_gap=result.gap,
                   labels=int(self.counts.sum()),counts=self.counts.tolist(),
                   true_gain=v,kl=kl(q,self.b),query_steps=self.steps)

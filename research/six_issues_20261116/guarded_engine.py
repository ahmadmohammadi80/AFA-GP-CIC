"""Generic finite-policy guarded AFA with incremental general-opponent scores.

No oracle mean is available to the selector. Input Q,H must be fixed before
feedback. Only mathematically zero influence edges are pruned (no tolerance).
The returned policy is a recommendation; deploy only when certified is True.
"""
import math
import numpy as np
from support_pruning import coefficients
from theory_and_guarantees import clipped_intersection, radius
from bounded_drift import bound as drift_bound

class GuardedEngine:
    def __init__(self,b,Q,H,delta=.05,threshold=.015,period=20,kappa=None,
                 prune=False,incremental=True,rho=None):
        b=np.asarray(b,float);Q=np.asarray(Q,float);H=np.asarray(H,float)
        if b.ndim!=1 or len(b)<2 or Q.ndim!=2 or H.ndim!=2 or not len(Q) or not len(H):
            raise ValueError('nonempty distributions required')
        if Q.shape[1]!=len(b) or H.shape[1]!=len(b):raise ValueError('shape')
        for x in (b,Q,H):
            if not np.all(np.isfinite(x)) or np.any(x<0) or not np.allclose(x.sum(axis=-1),1,rtol=0,atol=1e-12):
                raise ValueError('probability distributions required')
        if np.any(b<=0):raise ValueError('reference must have full support')
        if not 0<delta<1 or not np.isfinite(threshold) or not isinstance(period,int) or period<2:
            raise ValueError('confidence/threshold/period')
        if kappa is not None:
            if not np.isfinite(kappa) or kappa<0:raise ValueError('KL budget')
            logq=np.zeros_like(Q);np.log(Q,out=logq,where=Q>0)
            if np.any(np.sum(Q*(logq-np.log(b)),axis=1)>kappa+1e-12):raise ValueError('KL budget exceeded')
        if rho is not None and (not np.isfinite(rho) or rho<0):raise ValueError("known drift bound")
        self.rho=rho;self.now=0
        self.edges,self.A=coefficients(b,Q,H)
        self.absA=np.abs(self.A)
        self.active=np.flatnonzero(np.any(self.A!=0,axis=(0,1))) if prune else np.arange(len(self.edges))
        self.E=len(self.active);self.delta=delta;self.threshold=threshold;self.period=period
        self.incremental=incremental;self.n=np.zeros(len(self.edges),int);self.wins=self.n.copy()
        self.lo=np.full(len(self.edges),-1.);self.hi=-self.lo
        self.scores=-self.absA.sum(axis=-1);self.upper_scores=-self.scores.copy();self.labels=0;self.rr=0
        self.S=float(np.max(self.absA.sum(axis=-1)))
        self.Q=Q.copy();self.time_sums=np.zeros(len(self.edges))

    def bounds(self):
        # Frequent complete rebasing bounds accumulated rounding drift; tests
        # compare the resulting decisions with the dense path on shared data.
        if not self.incremental or self.labels%128==0:
            self.scores=np.sum(np.where(self.A>=0,self.A*self.lo,self.A*self.hi),axis=-1)
            self.upper_scores=np.sum(np.where(self.A>=0,self.A*self.hi,self.A*self.lo),axis=-1)
        # Conservative floating-point allowance, not an interval-arithmetic proof.
        pad=32*np.finfo(float).eps*(1+self.S)*(1+min(self.labels,128))
        lower=self.scores.min(axis=1)-pad
        upper=self.upper_scores.min(axis=1)+pad
        return lower,upper

    def next_edge(self):
        if not self.E:return None
        missing=self.active[self.n[self.active]==0]
        if len(missing):return int(missing[0])
        post=self.labels-self.E
        if (post+1)%self.period==0:
            e=int(self.active[self.rr%self.E]);self.rr+=1;return e
        lower,_=self.bounds();top=int(np.argmax(lower))
        worst=np.argmin(self.scores,axis=1)
        P=self.absA[np.arange(len(self.Q)),worst]*(self.hi-self.lo)/2/(np.sqrt(self.n)+.2)
        score=.7*P[top]+.3*P.max(axis=0)
        return int(self.active[np.argmax(score[self.active])])

    def observe(self,e,win):
        if e not in self.active or win not in (0,1):raise ValueError('edge/outcome')
        if self.rho is not None and self.now>self.labels:raise ValueError('sampling after calendar jump')
        self.n[e]+=1;self.wins[e]+=win;self.labels+=1
        self.time_sums[e]+=self.labels
        if self.rho is not None:
            self.advance(self.labels)
            return
        lo,hi=clipped_intersection(int(self.wins[e]),int(self.n[e]),self.delta,self.E)
        a=self.A[:,:,e]
        self.scores+=np.where(a>=0,a*(lo-self.lo[e]),a*(hi-self.hi[e]))
        self.upper_scores+=np.where(a>=0,a*(hi-self.hi[e]),a*(lo-self.lo[e]))
        self.lo[e]=lo;self.hi[e]=hi

    def advance(self,now):
        """Refresh ALL stale pairs under known drift; time may not go backwards.

        Online observations currently occur at consecutive integer times.
        advance() is also available for checking a later deployment time after
        sampling ends; collecting more labels after such a jump is rejected.
        """
        if self.rho is None:raise ValueError('advance requires known rho')
        if not np.isfinite(now) or now<self.now:raise ValueError('calendar time')
        self.now=now
        for e in self.active:
            self.lo[e],self.hi[e]=drift_bound(int(self.wins[e]),int(self.n[e]),
                self.time_sums[e],now,self.rho,self.delta,self.E)
        self.scores=np.sum(np.where(self.A>=0,self.A*self.lo,self.A*self.hi),axis=-1)
        self.upper_scores=np.sum(np.where(self.A>=0,self.A*self.hi,self.A*self.lo),axis=-1)

    def result(self):
        lower,upper=self.bounds();top=int(np.argmax(lower))
        all_seen=self.E==0 or np.all(self.n[self.active]>0)
        rate=0. if self.E==0 else (2*self.S*radius(int(min(self.n[self.active])),self.delta,self.E) if all_seen else 4.)
        return {'labels':self.labels,'policy_index':top,'policy':self.Q[top].tolist(),
                'certified':bool(lower[top]>self.threshold),
                'lower':float(lower[top]),'upper_best':float(max(upper)),
                'simple_regret_certificate':float(max(0,max(upper)-lower[top])),
                'rate_bound':float(rate) if self.rho is None else None,
                'confidence_mode':'stationary' if self.rho is None else 'known_bounded_drift',
                'calendar_time':self.now if self.rho is not None else self.labels,
                'active_edges':self.E,
                'counts':self.n.tolist()}

    def run(self,oracle,budget):
        if not isinstance(budget,int) or budget<0:raise ValueError('nonnegative integer budget')
        while self.labels<budget:
            if self.result()['certified']:break
            e=self.next_edge()
            if e is None:break
            self.observe(e,oracle(e,self.labels+1))
        return self.result()

"""AFA-CIC-JC: joint candidate/query/certificate finite-action cyclic preference prototype.
Research artifact; global scientific novelty is NOT claimed. Copyright (c) 2026 Ahmad Mohammadi.

No reward scalar or BTL is assumed. Query labels for unordered pair (i,j), i<j,
are independent Bernoulli with unknown fixed probability p_ij. W_ij=2*p_ij-1.
The time-uniform simultaneous confidence intervals support fully adaptive selection
of pairs, policy candidates, and arbitrary stopping times.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import betabinom
from dataclasses import dataclass


def kl(q,p):
    q=np.asarray(q,float); p=np.asarray(p,float)
    if np.any(q<0) or np.any(p<=0): return float('inf')
    mask=q>0
    return float(np.sum(q[mask]*np.log(q[mask]/p[mask])))


def all_edges(k):
    return [(i,j) for i in range(k) for j in range(i+1,k)]


def frontier_candidates(reference,kappa):
    p=np.asarray(reference,float)
    if np.any(p<=0) or abs(p.sum()-1)>1e-9: raise ValueError('reference needs full support')
    qs=[]
    for i in range(len(p)):
        target=np.zeros_like(p);target[i]=1.
        lo,hi=0.,1.
        for _ in range(60):
            mid=(lo+hi)/2
            q=(1-mid)*p+mid*target
            if kl(q,p)<=kappa:lo=mid
            else:hi=mid
        qs.append((1-lo)*p+lo*target)
    return np.array(qs)


def coeffs(q,b):
    # q^T W b = sum_{i<j} (q_i*b_j - q_j*b_i) W_ij
    return np.array([q[i]*b[j]-q[j]*b[i] for i,j in all_edges(len(b))],float)


def pair_probability_matrix(k,values):
    W=np.zeros((k,k))
    for (i,j),w in zip(all_edges(k),values):
        W[i,j]=w;W[j,i]=-w
    return W


def sign_cycle(W,a=0,b=1,c=2):
    return W[a,b]>0 and W[b,c]>0 and W[c,a]>0

@dataclass
class Certificate:
    reference: np.ndarray
    delta:float=.05

    def __post_init__(self):
        self.reference=np.array(self.reference,float)
        if not 0<self.delta<1: raise ValueError('delta range')
        self.edges=all_edges(len(self.reference))
        self.counts=np.zeros(len(self.edges),dtype=int)
        self.wins=np.zeros(len(self.edges),dtype=int)

    def observe(self,e,wins,n):
        if n<0 or wins<0 or wins>n:raise ValueError('counts')
        self.counts[e]+=int(n);self.wins[e]+=int(wins)

    def interval(self):
        M=len(self.edges); n=self.counts
        center=np.divide(self.wins,n,out=np.full(M,.5),where=n>0)
        rad=np.ones(M)
        mask=n>0
        # 2 exp(-2 n r_p^2)=6 delta / (pi^2 M n^2)
        # delta simultaneously over all edges and all n >=1 via sum 1/n^2=pi^2/6.
        rad[mask]=np.sqrt(np.log(np.pi**2*M*n[mask].astype(float)**2/(3*self.delta))/(2*n[mask]))
        return 2*np.maximum(0,center-rad)-1, 2*np.minimum(1,center+rad)-1

    def lower(self,q):
        a=coeffs(q,self.reference);lo,hi=self.interval()
        return float(np.sum(np.where(a>=0,a*lo,a*hi)))

    def upper(self,q):
        a=coeffs(q,self.reference);lo,hi=self.interval()
        return float(np.sum(np.where(a>=0,a*hi,a*lo)))

    def empirical(self,q):
        a=coeffs(q,self.reference)
        phat=np.divide(self.wins,self.counts,out=np.full(len(a),.5),where=self.counts>0)
        return float(a@(2*phat-1))

    def certify(self,qs,kappa):
        feas=[q for q in qs if kl(q,self.reference)<=kappa+1e-9]
        if not feas:return None
        lbs=np.array([self.lower(q) for q in feas])
        i=int(np.argmax(lbs))
        return feas[i].copy() if lbs[i]>0 else None

    def query(self,qs,method='joint',fixed_candidate=None,block=1):
        if method in ('voi','voi_fixed'):
            # Exact finite-horizon one-block expected certificate-value update.
            # All candidate certificates are recomputed counterfactually for each possible
            # win count, so query planning explicitly accounts for policy switching.
            A=np.array([coeffs(q,self.reference) for q in ([fixed_candidate] if method=='voi_fixed' else qs)])
            lo,hi=self.interval()
            old_values=np.where(A>=0,A*lo,A*hi).sum(axis=1)
            now=old_values.max()
            # The Beta-binomial is the exact finite-batch posterior predictive under Beta(1,1) priors.
            # It is used ONLY for query priority; the safety test is frequentist.
            potentials=[]
            for e in range(len(self.edges)):
                new_n=int(self.counts[e]+block)
                xs=np.arange(block+1)
                new_phat=(self.wins[e]+xs)/new_n
                r=np.sqrt(np.log(np.pi**2*len(self.edges)*new_n**2/(3*self.delta))/(2*new_n))
                new_lo=2*np.maximum(0,new_phat-r)-1
                new_hi=2*np.minimum(1,new_phat+r)-1
                old_part=np.where(A[:,e]>=0,A[:,e]*lo[e],A[:,e]*hi[e])
                new_part=np.where(A[:,e,None]>=0,A[:,e,None]*new_lo, A[:,e,None]*new_hi)
                new_best=(old_values[:,None]-old_part[:,None]+new_part).max(axis=0)
                prob=betabinom.pmf(xs,block,1+self.wins[e],1+self.counts[e]-self.wins[e])
                # Direct value of information for acceptance, with a small preference for
                # increasing expected best certificate even if crossing 0 not yet feasible.
                utility=float(prob@np.maximum(new_best,0)) + .2*float(prob@new_best-now)
                potentials.append(utility)
            return int(np.argmax(potentials))
        if method=='uniform':return int(np.argmin(self.counts))
        if method=='variance':
            phat=np.divide(self.wins,self.counts,out=np.full(len(self.counts),.5),where=self.counts>0)
            return int(np.argmax(phat*(1-phat)/(self.counts+1)))
        if method=='fixed':
            if fixed_candidate is None:raise ValueError('fixed candidate missing')
            q=fixed_candidate
        elif method=='joint':
            # Policy, sample allocation and rejection threshold are coupled:
            # prioritize certificate candidates by empirical gain / current uncertainty.
            est=np.array([self.empirical(q) for q in qs])
            lower=np.array([self.lower(q) for q in qs])
            upper=np.array([self.upper(q) for q in qs])
            # Candidate must have some plausible certificate upside; balance uncertain alternatives.
            reachable=np.maximum(0,upper)
            candidate_scores=est - .2*(upper-lower) + .02*reachable
            q=qs[int(np.argmax(candidate_scores))]
        else:raise ValueError(method)
        a=np.abs(coeffs(q,self.reference))
        n=self.counts
        # Predicted decrease in simultaneous Hoeffding width after a batch, only using counts.
        M=len(self.edges)
        def rad_at(counts):
            x=np.maximum(counts,1).astype(float)
            r=np.sqrt(np.log(np.pi**2*M*x*x/(3*self.delta))/(2*x))
            return 2*np.minimum(1,r)
        widths=rad_at(n)-rad_at(n+block)
        scores=a*np.maximum(0,widths)
        if scores.max()<1e-12: scores=a/(n+1)
        return int(np.argmax(scores))


def generated_problem(seed,k=5):
    rng=np.random.default_rng(seed)
    b=np.array([.42,.20,.17,.12,.09])
    if k!=5:raise ValueError('k=5 only')
    W=np.zeros((k,k))
    E=all_edges(k)
    vals=np.array([.66,-.70,.48,-.1, .68,.10,-.18, .25,.22, .40])
    # edges lex order (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
    vals=np.clip(vals + rng.normal(0,.09,len(vals)),-.90,.90)
    for (i,j),v in zip(E,vals):W[i,j]=v;W[j,i]=-v
    assert sign_cycle(W,0,1,2)
    return b,W


def run_trial(seed,method='joint',budget=1500,pilot=25,block=20,kappa=.25,delta=.05):
    b,W=generated_problem(seed)
    qs=frontier_candidates(b,kappa)
    cs=Certificate(b,delta)
    rng=np.random.default_rng(seed+834711)
    for e,(i,j) in enumerate(cs.edges):
        cs.observe(e,int(rng.binomial(pilot,(1+W[i,j])/2)),pilot)
    fixed=qs[int(np.argmax([cs.empirical(q) for q in qs]))]
    t=pilot*len(cs.edges)
    accepted=None
    while t<budget:
        accepted=cs.certify(qs,kappa)
        if accepted is not None:break
        batch=min(block,budget-t)
        e=cs.query(qs,method,fixed,block=batch)
        i,j=cs.edges[e]
        cs.observe(e,int(rng.binomial(batch,(1+W[i,j])/2)),batch)
        t+=batch
    if accepted is None:accepted=cs.certify(qs,kappa)
    q=b if accepted is None else accepted
    return {'seed':seed,'method':method,'comparisons':t,'certified':accepted is not None,
            'true_gain':float(q@W@b),'returned_kl':kl(q,b),
            'max_lower_bound':max(cs.lower(qq) for qq in qs),
            'covers':bool(np.all([cs.interval()[0][e]-1e-12 <= W[i,j] <= cs.interval()[1][e]+1e-12 for e,(i,j) in enumerate(cs.edges)])),
            'selected_action':None if accepted is None else int(np.argmax(accepted-b)),
            'cycle':bool(sign_cycle(W)),
            'counts':cs.counts.tolist()}

"""Primal / dual KL-robust preference certificate.

For b_i>0, sum(b)=1, C=[lo,hi], kappa>=0:
  V = max_{q simplex, D_KL(q||b)<=kappa} min_{s in C}(q-b)^T s.

For any feasible q:  P(q) <= V.
For any s in C, eta>0: V <= U(s,eta) = eta*kappa + eta*log sum b_i exp(s_i/eta) - b^T s.
Thus a feasible q and (s,eta) give a *mathematically auditable*
primal/dual gap even when numerical optimizers fail to reach optimum.

This code uses ordinary floating point. For formal computer-assisted proof of
numeric tolerances an interval-arithmetic implementation would be needed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def check(b, lo, hi, kappa):
    b=np.asarray(b,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float)
    if b.ndim!=1 or len(b)<2 or lo.shape!=b.shape or hi.shape!=b.shape:
        raise ValueError('b, lo, hi must be 1D arrays of same length >=2')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError('nonfinite input')
    if np.any(b<=0) or abs(b.sum()-1)>1e-10:
        raise ValueError('b must be a full-support probability vector')
    if np.any(lo>hi) or np.any(lo < -1.-1e-10) or np.any(hi>1.+1e-10):
        raise ValueError('bounds must define valid [-1,1] intervals')
    if not np.isfinite(kappa) or kappa<0:
        raise ValueError('kappa must be finite, nonnegative')
    return b,lo,hi


def kl(q,b):
    q=np.asarray(q,float); b=np.asarray(b,float)
    if np.any(q < 0) or np.any(b <= 0): return float('inf')
    ix=q>0
    return float(np.sum(q[ix]*np.log(q[ix]/b[ix])))


def lower(q,b,lo,hi):
    d=np.asarray(q)-np.asarray(b)
    return float(np.sum(np.where(d>=0,d*lo,d*hi)))


def dual(s, eta, b, kappa):
    if eta<=0 or not np.isfinite(eta):raise ValueError('eta must be >0')
    s=np.asarray(s,float); b=np.asarray(b,float)
    val=eta*kappa + eta*logsumexp(np.log(b)+s/eta)-float(b@s)
    return float(val)


def softmax_policy(s,eta,b):
    z=np.log(b)+np.asarray(s)/eta
    q=np.exp(z-logsumexp(z))
    return q


def mix_to_kl(q,b,kappa):
    q=np.asarray(q,float); b=np.asarray(b,float)
    if kappa<=0: return b.copy()
    if kl(q,b)<=kappa*(1-1e-10):return q.copy()
    low,high=0.,1.
    for _ in range(75):
        mid=(low+high)/2
        if kl(b+mid*(q-b),b)<=kappa*(1-1e-10):low=mid
        else:high=mid
    return b+low*(q-b)


@dataclass
class Outcome:
    q:np.ndarray
    primal:float
    dual_upper:float
    gap:float
    kl:float
    s_dual:np.ndarray
    eta:float
    iterations:int
    success:bool
    diagnostics:str


def solve(b,lo,hi,kappa,tolerance=2e-4,maxiter=2400):
    b,lo,hi=check(b,lo,hi,kappa)
    if kappa==0:
        return Outcome(b.copy(),0.,0.,0.,0.,(lo+hi)/2,0.,0,True,'zero KL: only q=b feasible')
    # Exact zero certificate: a constant adversarial score c belongs to all
    # confidence intervals. Hence (q-b)^T(c*1)=0 for ALL policies.
    if float(np.max(lo)) <= float(np.min(hi)):
        c=(float(np.max(lo))+float(np.min(hi)))/2
        return Outcome(b.copy(),0.,0.,0.,0.,np.full_like(b,c),0.,0,True,
                       'all confidence intervals share a common value: exact optimum zero')

    # L-BFGS-B minimizes the *jointly convex* dual in s and eta.
    # Derivatives: partial_s D=q_eta(s)-b; d_eta D=kappa+logZ-E_q(s)/eta.
    k=len(b)
    def dual_with_jac(x):
        s=x[:k]; eta=x[-1]
        z=np.log(b)+s/eta
        logz=logsumexp(z)
        q=np.exp(z-logz)
        obj=eta*kappa+eta*logz-float(b@s)
        ds=q-b
        deta=kappa+logz-float(q@s)/eta
        return obj, np.r_[ds,deta]
    # A feasible dual value is ALWAYS an upper bound, regardless of convergence.
    initial_eta=max(.03,min(10., .30/np.sqrt(kappa)))
    eta_upper=max(30., 5./np.sqrt(kappa))
    starts=[np.r_[(lo+hi)/2,initial_eta], np.r_[np.clip(np.zeros(k),lo,hi),initial_eta],np.r_[lo,initial_eta]]
    opts=[]
    for x0 in starts:
        r=minimize(dual_with_jac,x0,jac=True,method='L-BFGS-B',
            bounds=list(zip(lo,hi))+[(1e-10,eta_upper)],
            options={'maxiter':maxiter,'ftol':1e-14,'gtol':1e-11,'maxls':45})
        if np.all(np.isfinite(r.x)) and np.isfinite(r.fun):
            opts.append(r)
    if not opts:raise RuntimeError('dual search did not return finite point')
    r=min(opts,key=lambda x: x.fun)
    sstar=r.x[:-1]; etastar=float(r.x[-1]); ub=dual(sstar,etastar,b,kappa)
    # Certified primal feasibility from explicit projection along b->softmax.
    q0=mix_to_kl(softmax_policy(sstar,etastar,b),b,kappa)
    candidates=[q0,b.copy()]
    # The primal problem is minimize convex -lower, constrained by KL.
    # SLSQP yields feasible candidates but does not itself certify global optimum.
    def objective(q):
        return -lower(q,b,lo,hi)
    def gradient(q):
        d=q-b
        return -np.where(d>=0,lo,hi)
    constr=[{'type':'eq','fun':lambda q:float(np.sum(q)-1.), 'jac':lambda q:np.ones(k)},
            {'type':'ineq','fun':lambda q:float(kappa-kl(q,b)),
             'jac':lambda q:-np.log(np.maximum(q,1e-14)/b)-1.}]
    for start in [q0, b.copy()]:
        rs=minimize(objective,start,jac=gradient,method='SLSQP',
            constraints=constr,bounds=[(1e-14,1.)]*k,
            options={'maxiter':maxiter,'ftol':2e-12})
        if np.all(np.isfinite(rs.x)) and np.max(np.abs(rs.x))<=1.0000001 and abs(np.sum(rs.x)-1)<1e-7:
            trial=np.maximum(rs.x,0.)
            trial=trial/trial.sum()
            candidates.append(mix_to_kl(trial,b,kappa))
    chosen=max(candidates,key=lambda q: lower(q,b,lo,hi))
    l=lower(chosen,b,lo,hi)
    kd=kl(chosen,b)
    if not (kd<=kappa+1e-10 and abs(sum(chosen)-1)<1e-9 and np.all(chosen>=0)):
        raise ArithmeticError('primal feasibility guard failed')
    raw_gap=ub-l
    if raw_gap < -1e-8:
        raise ArithmeticError('weak duality violated; numeric optimizer unstable')
    gap=max(0.,raw_gap)  # tiny negative round-off is checked above
    return Outcome(q=chosen,primal=l,dual_upper=ub,gap=gap,kl=kd,
                   s_dual=sstar,eta=etastar,iterations=int(r.nit),
                   success=bool(gap<=tolerance),diagnostics=('gap meets tolerance' if gap<=tolerance else 'optimization unresolved; returned valid brackets'))

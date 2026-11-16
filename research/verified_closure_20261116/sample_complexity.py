"""Explicit finite-sample certification bound for uniform and forced exploration.

Proof: for Y in [-1,1], the time-uniform Hoeffding/union radius
  r(n)=min(2,sqrt(2 log(A n^2)/n)), A=pi^2 K/(3delta).
If K>=2, delta<1 and 0<eps<=1, then
  n>=ceil(16/eps^2 * log(16*A/eps^4))
implies r(n)<=eps. (Proof in PROOF_CLOSURE_FA.md.)

For any KL-feasible q* with true gap G(q*)>=gamma+Delta, the empirical
rectangular certificate satisfies lower(q*)>=gamma+Delta-2r sqrt(2kappa)
on the simultaneous coverage event. With certified global solver gap <=Delta/4,
setting eps=min(1,Delta/(8 sqrt(2kappa))) ensures acceptance at level gamma.
Uniform sample budget K*n; in a block controller that forces least-count
arm once every m steps, a conservative worst-case bound is m*K*ceil(n/B)*B.
This is NOT optimal sample complexity of adaptive sampling.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import pi,sqrt,log,ceil,isfinite


def confidence_radius(n,k,delta):
    if not isinstance(n,int) or n<0 or not isinstance(k,int) or k<2 or not 0<delta<1:
        raise ValueError('invalid n, K or delta')
    return 2.0 if n==0 else min(2.,sqrt(2./n*log(pi*pi*k*n*n/(3*delta))))


def n_for_radius(k,delta,eps):
    if not isinstance(k,int) or k<2 or not 0<delta<1 or not isfinite(eps) or not 0<eps<=1:
        raise ValueError('K>=2, delta in (0,1), eps in (0,1] required')
    a=pi*pi*k/(3*delta)
    val=16/eps**2 * log(16*a/eps**4)
    if not isfinite(val) or val>1e16:raise OverflowError('theoretical budget too large for floating-point rounding')
    n=ceil(val)+2 # roundoff slack, exact theorem holds for n>=real expression
    assert confidence_radius(n,k,delta)<=eps*(1+1e-12)
    return int(n)


@dataclass(frozen=True)
class Bound:
    per_arm:int
    uniform_labels:int
    forced_exploration_labels:int
    radius_target:float
    certified_solver_gap_allowed:float
    model:str


def safe_certificate_budget(k,delta,kappa,Delta,block=25,explore_every=5):
    if not isfinite(kappa) or not isfinite(Delta) or kappa<=0 or Delta<=0:
        raise ValueError('positive finite kappa and margin Delta required')
    if not isinstance(block,int) or not isinstance(explore_every,int) or block<1 or explore_every<1:
        raise ValueError('positive integer block/explore_every required')
    eps=min(1.,Delta/(8*sqrt(2*kappa)))
    n=n_for_radius(k,delta,eps)
    return Bound(n,k*n,explore_every*k*ceil(n/block)*block,eps,Delta/4,
                 'Stationary i.i.d. preference labels, full-support fixed b, true margin Delta, '
                 'time-uniform confidence and per-call primal-dual gap <= Delta/4')

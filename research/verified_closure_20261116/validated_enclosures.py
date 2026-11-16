"""High-precision interval enclosure of a finite KL certificate.

Assumptions: all supplied finite decimal strings are interpreted as exact
rational numbers. We recompute the final coordinate of b and q to enforce
*exact* simplex membership, and conservatively mix q toward b until an
interval upper bound on KL lies below the rational budget.

The primal bound is evaluated in exact Fraction arithmetic; the dual upper
bound uses mpmath.iv with 80 decimal digits and an explicit outward enclosure.
This is independently checkable numerical evidence, not a proof of the
underlying interval library's implementation or end-to-end LLM alignment.
"""
from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass
from math import isfinite
import mpmath as mp


def fr(x):
    if isinstance(x, Fraction): return x
    if isinstance(x, str): return Fraction(x)
    if not isfinite(float(x)): raise ValueError('finite decimal values required')
    return Fraction(repr(float(x)))


def simplex(vals):
    vals = [fr(x) for x in vals]
    if len(vals)<2: raise ValueError('at least two actions required')
    vals[-1] = 1-sum(vals[:-1])
    if min(vals)<0: raise ValueError('negative simplex coordinate')
    return vals


def from_mpf_tuple(t):
    sign, man, exp, _ = t
    man = -man if sign else man
    return Fraction(man * (2**exp)) if exp>=0 else Fraction(man, 2**(-exp))


def lower_and_upper_interval(x):
    return from_mpf_tuple(x._mpi_[0]), from_mpf_tuple(x._mpi_[1])


def interval(x):
    x=fr(x)
    return mp.iv.mpf(x.numerator)/mp.iv.mpf(x.denominator)


def kl_interval(q,b):
    r=mp.iv.mpf(0)
    for qi,bi in zip(q,b):
        if qi>0:r += interval(qi)*mp.iv.ln(interval(qi)/interval(bi))
    return lower_and_upper_interval(r)


def lower_exact(q,b,L,U):
    return sum((qi-bi)*(li if qi>=bi else ui) for qi,bi,li,ui in zip(q,b,L,U))


def dual_upper_interval(s,eta,b,kappa):
    # Exact shift of logits limits cancellation and over/underflow.
    m=max(s)
    total=mp.iv.mpf(0)
    for si,bi in zip(s,b):
        total+=interval(bi)*mp.iv.exp(interval((si-m)/eta))
    val=interval(eta*kappa+m)+interval(eta)*mp.iv.ln(total)
    for si,bi in zip(s,b):val-=interval(bi*si)
    return lower_and_upper_interval(val)


@dataclass(frozen=True)
class Enclosure:
    primal_lower_decimal: str
    dual_upper_decimal: str
    gap_upper_decimal: str
    kl_upper_decimal: str
    kappa_decimal: str
    exact_feasible: bool
    optimality_tol_pass: bool
    mixing_iterations: int
    probability_vector_decimals: tuple[str,...]
    policy_vector_rational: tuple[str,...]
    reference_vector_rational: tuple[str,...]


def certify(b,q,L,U,kappa,s,eta,tol='0.0002',digits=80):
    if digits<32: raise ValueError('digits must be at least 32')
    mp.iv.dps=digits
    b=simplex(b);q=simplex(q)
    L=[fr(x) for x in L];U=[fr(x) for x in U]
    s=[fr(x) for x in s];eta=fr(eta);kappa=fr(kappa);tol=fr(tol)
    if len(L)!=len(b) or len(U)!=len(b) or len(s)!=len(b):raise ValueError('dimension mismatch')
    if min(b)<=0 or min(q)<0 or any(li>ui for li,ui in zip(L,U)) or any(si<li or si>ui for li,ui,si in zip(L,U,s)):
        raise ValueError('invalid b/q bounds or dual witness')
    if eta<0 or kappa<0 or tol<0:raise ValueError('invalid scalars')
    if kappa==0:q=b.copy()
    klhi=Fraction(0) if q==b else kl_interval(q,b)[1]
    mix=0
    while klhi>kappa and mix<100:
        # 0.9 shrink is conservative, but this case should occur only near budget.
        q=[bi+Fraction(9,10)*(qi-bi) for qi,bi in zip(q,b)]
        klhi=Fraction(0) if q==b else kl_interval(q,b)[1]
        mix+=1
    if klhi>kappa:raise ArithmeticError('failed to establish KL feasibility')
    p=lower_exact(q,b,L,U)
    if kappa==0:
        up=Fraction(0)  # singleton KL ball
    elif eta==0:
        up=max(s)-sum(bi*si for bi,si in zip(b,s)) # eta->0 conjugate limit
    else:
        up=dual_upper_interval(s,eta,b,kappa)[1]
    if up<p:raise ArithmeticError('numerically invalid weak-duality enclosure')
    gap=up-p
    # Decimal strings with enough digits are for display, not further proof.
    fmt=lambda x:str(mp.nstr(mp.mpf(x.numerator)/x.denominator,40))
    return Enclosure(fmt(p),fmt(up),fmt(gap),fmt(klhi),fmt(kappa),
                     True,gap<=tol,mix,tuple(fmt(x) for x in q),
                     tuple(str(x) for x in q),tuple(str(x) for x in b))

"""Fail-closed wrapper: policy used for certification is the rationalized output.

This wrapper enforces: finite-action KL feasibility, exact rational primal
lower bound, interval-computed dual upper bound, and specified gap tolerance.
Call it instead of trusting optimizer.success or an unrounded float KL value.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from certified_solver import solve
from validated_enclosures import certify, Enclosure


@dataclass(frozen=True)
class VerifiedOutcome:
    audit: Enclosure
    policy_exact_rationals: tuple[str,...]
    solver_iterations: int
    declared_model: str


def solve_verified(b,lo,hi,kappa,tolerance='0.0002'):
    raw=solve(b,lo,hi,kappa,tolerance=float(tolerance))
    proof=certify(b,raw.q,lo,hi,kappa,raw.s_dual,raw.eta,tol=tolerance)
    if not proof.optimality_tol_pass:
        raise ArithmeticError('primal-dual interval gap above requested tolerance: '+proof.gap_upper_decimal)
    if not proof.exact_feasible:raise ArithmeticError('KL audit failed')
    assert sum(Fraction(x) for x in proof.policy_vector_rational)==1
    assert sum(Fraction(x) for x in proof.reference_vector_rational)==1
    return VerifiedOutcome(proof,proof.policy_vector_rational,raw.iterations,
        'Fixed-support finite-action KL ball; stationary bounded pairwise observations')

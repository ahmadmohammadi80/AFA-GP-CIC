# AFA-GP-CIC — exact certificate-influence pruning: theorem and novelty gate

## Operator (one new candidate in this run)
For a **fixed finite** policy family Q, opponent family H, and baseline b, let
A[q,h,(i,j)] = (q_i-b_i)h_j-(q_j-b_j)h_i. Query only the union of edges
S={e: exists q,h with A[q,h,e] != 0}. All omitted edges are ignored in the
lower certificate; confidence allocation is delta/|S| on queried edges.

## Theorem (exactness and a limited sample allocation separation)
For every skew-symmetric W and every vector of per-edge lower/upper bounds,
the entire vector of candidate robust lower certificates is identical after
removing all e outside S. Moreover, any fixed-round-robin design querying
all E edges and any fixed-round-robin design querying only S require exactly
E*n and |S|*n labels respectively to collect n observations per relevant
edge (up to an initial incomplete cycle). The deterministic factor E/|S|
is a separation **only against all-edge uniform allocation**, not against
edge-aware adaptive learning.

Proof: the robust gain for each (q,h) is sum_e A[q,h,e] W_e. If e not in S,
the coefficient vanishes for every q,h, so its contribution to both gain
and box-confidence lower bound is zero. The query count is a direct counting
identity. Union confidence bounds on S suffice for arbitrary adaptive stopping
because inactive edges never appear in a certificate.

**Important limitations:** This does not yield a nontrivial minimax lower
bound against an optimal support-aware adaptive learner. If Q or H changes,
S must be recomputed and previous confidence allocation tracked; for
continuous unrestricted Q,H the support is generally dense. Thus this
operator is not yet a defensible globally independent novelty claim.

## Matched-information/budget gate
- All methods receive the same b, Q, H, KL radius, threshold, and initial
  edge outcomes. All methods have the same n label budget.
- Our operator and an all-edge round-robin both know the full A tensor,
  so the separation is a deliberately **weak** comparator, not a strong
  state-of-the-art baseline.
- Robust Nash Alignment (arXiv:2610.00715) optimizes a robust Nash
  distribution under preference uncertainty; it does not directly solve
  this fixed-baseline anytime certificate stopping objective. A faithful
  translation needs a shared oracle and terminal certificate.
- Nash Learning (Munos et al., ICML 2024) uses preference-based Nash
  optimization, not this safety stopping criterion.
- POSTPI (Tse et al., RLC 2025) targets high-confidence policy improvement;
  its observation/feedback protocol must be matched before claiming a
  difference in labels.
- Nearly Optimal Active Preference Learning (Zhao & Jun, 2026) already
  develops instance-dependent active label complexity; support pruning
  may be a special case and must not be advertised as new without a
  direct reduction check.

## Tests and falsification gate
`test_support_pruning.py` compares the full and pruned certificates at
machine precision on multiple dimensions. `support_pruning.py` constructs
a 20-action sparse-support witness with four active edges out of 190,
and runs both policies at the same 10,000-label budget using seeded
independent per-edge streams. CI artifacts preserve exact outputs.
No empirical result is asserted here until the CI run is inspected.

## Next
Derive a lower bound against a **support-aware**, full-information
adaptive oracle; if this fails, reject this operator as the central novelty
and move to a genuine stopping-objective-specific mechanism.

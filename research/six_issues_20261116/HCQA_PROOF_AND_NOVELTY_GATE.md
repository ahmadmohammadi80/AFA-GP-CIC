# HCQA: finite-horizon active-certificate allocation — verified scope

## One algorithmic contribution under test
**Horizon-Certified Query Allocation (HCQA)** replaces a one-step certificate-width query rule by a finite-integer-budget, multi-active-row minimax allocation with admissible branch-and-bound. The rows represent active policy/opponent certificate constraints. This is a deterministic planning oracle and NOT a claim of novel global preference-learning optimality.

Given a fixed nonnegative row-edge coefficient matrix A, current edge counts n_e>=1, c_e>=0 and remaining integer B, solve
F_B = min_{x in Z_+^E, sum x=B} max_r sum_e A_{re} c_e / sqrt(n_e+x_e).
Each edge label costs one unit. All methods see the same A, n, c and B.

## Theorem (exactness and safe pruning)
For a partial assignment x_1,...,x_{i-1} and remaining R labels, let
LB = max_r [sum_{e<i} A_re c_e/sqrt(n_e+x_e) + sum_{e>=i} A_re c_e/sqrt(n_e+R)].
Every feasible completion has objective >= LB because no unassigned edge can receive more than R labels. Consequently, pruning a branch only when LB exceeds the best feasible incumbent preserves the global optimum. Exhaustive enumeration of remaining integer allocations at leaves therefore returns F_B exactly. Complexity in the worst case is O(number_of_rows * E * binomial(B+E-1,E-1)); the algorithm is not scalable to large E without approximation.

**Important:** the theorem is deterministic, assumes known fixed A and radius law c/sqrt(n). It does not cover Beta-mixture or Bernoulli-KL intervals directly, data-dependent row screening, stochastic stopping time, or true label savings.

## Reproducible test
Run `python -m unittest discover -s research/six_issues_20261116 -p 'test_horizon_certificate.py' -v`
and `python research/six_issues_20261116/horizon_certificate.py`.
Local standalone run of equivalent implementation: 80/80 random cases matched exhaustive enumeration. Seed 27 yielded F_B=3.5666662658385695 vs one-step greedy 3.656819484326406 with B=6, A fixed as test fixture, x=[2,2,1,1]. This is a **constructed synthetic width counterexample**, not statistical evidence on preference learning.

## Matched comparison protocol, not yet completed
For Robust Nash Alignment, Nash Learning, POSTPI and active preference learning, use identical base policy b, KL radius, policy set Q, opponent set H, pairwise preference oracle and seeds; provide each method the same observation history and candidate edges at time zero, count one preference query as one label, match total budget B and decision frequency, and report anytime-certified positive gains and false certificates using one shared external evaluator. Methods optimizing a different Nash or regret target must be labeled **adapted**, not treated as native implementations. A fair comparator must also be allowed to use known A and active rows, otherwise HCQA receives privileged information.

Closest methodological risk: optimal experimental design, budgeted pure exploration, convex sample allocation and active learning already contain finite-horizon resource allocation ideas. Exact branch-and-bound is not by itself evidence of independent novelty.

## Gate
Formal finite-horizon oracle theorem: proven in stated deterministic scope.
Unit tests: check CI for this commit.
Comparison with native literature methods: OPEN.
Significant label-cost improvement: OPEN.
Independent, meaningful scientific novelty: NOT YET ESTABLISHED.

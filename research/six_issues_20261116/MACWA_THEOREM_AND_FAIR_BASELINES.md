# MACWA: exact one-step minimax certificate-width allocation

For a finite active set of certificate coefficient rows A, define U(n)=max_r sum_e |A[r,e]| radius(n[e]). MACWA chooses the next queried edge e minimizing U(n+unit_e). This differs from greedily querying the edge with largest individual uncertainty.

## Theorem and proof
Set w_r=sum_e |A[r,e]|radius(n[e]) and d_e=radius(n[e])-radius(n[e]+1). If radii are nonincreasing, d_e>=0. Then U(n+unit_e)=max_r(w_r-|A[r,e]|d_e). Only the e-th term changes, proving the identity. Enumerating all e and choosing the smallest objective exactly solves the one-step minimax width problem. Since every row's width cannot increase, U(n+unit_e)<=U(n). Complexity O(number_active_rows * number_edges), and memory O(number_active_rows + number_edges).

The theorem is only about deterministic widths, NOT stopping-time optimality, true lower-bound monotonicity, or a global sample complexity separation. The radius used is capped at 1 and based on the all-n Hoeffding bound; nonincreasingness is necessary and should be tested over the intended range.

## Evidence
The source file certificate_allocation.py and test_certificate_allocation.py implement the formula and an independent full recomputation oracle. Local Python unittest 3/3 passed, including 300 randomized cases comparing every candidate cost and argmin. GitHub Actions confirmation remains a separate requirement.

## Matched-information / matched-budget prior-art gate
All methods must see the same baseline b, KL cap, candidate policies Q, opponent set H, pairwise Bernoulli oracle, initial labels, time-uniform confidence intervals and total count of pairwise labels (including initialization). Use shared edge-indexed random streams and matched seeds. Allow all methods to adapt queries and stopping; score certification probability, false certification, label cost, and wall time separately.

Robust Nash Alignment (arxiv.org/abs/2610.00715): permit the same uncertainty set; its native Nash objective differs, so common safe-certification stopping is an explicitly adapted wrapper.
Nash Learning from Human Feedback (proceedings.mlr.press/v235/munos24a.html): same cyclic game, but equilibrium and safe baseline improvement are different targets.
POSTPI (rlj.org/papers/tse2025high): compare against high-confidence policy improvement with the same baseline, KL cap, and valid adaptive stopping.
Active preference learning (arxiv.org/abs/2402.10500; proceedings.mlr.press/v267/feng25g.html): allow the same adaptive edge selection, intervals and budgets, and include a strong greedy certificate-width baseline.

Open novelty gate: demonstrate a reproducible improvement against a strong equally informed adaptive comparator, ideally with a family-specific lower bound. One-step exactness alone does not establish independent novelty.

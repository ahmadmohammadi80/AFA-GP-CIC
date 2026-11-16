# Margin-aware horizon certificate allocation: scoped novelty gate

This is **one candidate extension**, not an established globally novel algorithm. HCQA minimizes the worst certificate width; this oracle instead maximizes the **best policy's worst-opponent predicted lower certificate** under the same integer label budget, using fixed empirical centers. It may prioritize a nearly-certifiable policy over the row with largest width.

For fixed policy-opponent scores m[q,h], nonnegative sensitivity weights a[q,h,e], positive edge counts n[e], nonnegative scale c[e], and integer B, define
G(B)=max_{x in Z_+^E, sum x=B} max_q min_h {m[q,h]-sum_e a[q,h,e] c[e]/sqrt(n[e]+x[e])}.

**Exact finite-budget theorem:** exhaustive enumeration of all weak compositions of B into E parts and evaluation of the displayed score returns G(B), and a feasible x maximizing it. Proof: every feasible integer allocation appears exactly once in the recursion, and the evaluation computes its objective exactly. There are binomial(B+E-1,E-1) allocations; runtime O(Q H E binomial(B+E-1,E-1)), memory O(E). This is a deterministic optimization theorem, NOT an original concentration theorem, a bound on true label complexity, or evidence of superiority over existing optimal experimental design.

**Conditional certificate corollary:** If the same simultaneous anytime event already established in the repository holds for every edge, policy and opponent, then for any adaptively selected q and stopping time, its *actual observed* lower confidence certificate remains conservative. Planning with frozen m does not itself establish future expected gain or guarantee that the planned lower score is achieved after collecting random labels. The planner must be rerun on actual data before issuing any certificate.

**Reproducibility:** python -m unittest discover -s research/six_issues_20261116 -p 'test_margin_horizon.py' -v. Three local tests passed, including 100 seeded small integer instances checked against an independently enumerated oracle. This verifies code correctness, not performance on preference feedback.

**Matched-information/budget comparison gate:** all algorithms must see the same baseline b, KL constraint, policies Q, opponents H, current observed edge counts and sample means, confidence sequence, sensitivity tensor, eligible queries, per-edge Bernoulli outcome streams, label budget B and decision opportunities. Count every preference query, including initialization; evaluate after actual sampled labels with one common anytime-valid certificate and report both success and label cost. For Robust Nash Alignment, Nash Learning, POSTPI and active preference learning, native objectives differ; report native and adapted tasks separately. An active greedy and HCQA baseline must receive the same scores and sensitivity tensor. No native method comparisons have been executed here.

**Status:** exact deterministic oracle implemented and locally tested; GitHub CI must be checked separately. Meaningful distinct scientific novelty, statistical improvement, and full literature comparator gate remain OPEN.

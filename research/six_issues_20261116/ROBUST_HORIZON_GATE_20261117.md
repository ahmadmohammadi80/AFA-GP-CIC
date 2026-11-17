# Robust finite-horizon certificate-crossing: conditional theorem and falsification gate (2026-11-17)

## One algorithm, not a novelty claim
Given a deterministic certificate predicate C(n,s), budget B, and rectangular edge-wise intervals [l_e,u_e] for Bernoulli parameters, solve
V_0(z)=C(z); V_t(z)=1 if C(z)=1; otherwise
V_t(z)=max_e min_{p in [l_e,u_e]} [(1-p)V_{t-1}(z_e^0)+p V_{t-1}(z_e^1)].
The inner minimum is at l_e if V(success)>=V(failure), and u_e otherwise. Select the smallest-index maximizing edge. Code: robust_horizon_crossing_20261117.py.

## Theorem (rectangular robust reachability, scoped)
If nature may choose a conditional Bernoulli probability in [l_e,u_e] at EACH future history, after observing the selected edge, then V_B equals the max-min probability of reaching C within B steps among all adaptive query policies. If the actual conditional probability lies in the corresponding interval at every history, the returned policy's true crossing probability is at least V_B. Proof: at t=0 the terminal indicator is exact. At t>0, the selected edge and its two outcomes partition all next histories. Rectangularity permits independent minimization of the conditional probability at every next history. The affine expectation is minimized at an interval endpoint. Backward induction proves the recursion and the claimed lower bound.

IMPORTANT: This theorem does NOT follow for a single unknown FIXED p_e constrained to an interval (nonrectangular coupling across histories); in that setting the recursion can be conservative, not exact. Nor does it guarantee validity of C: simultaneous anytime-valid intervals for the preference parameters and post-selection-safe certification must be supplied separately. If interval coverage holds with probability 1-delta, any unconditional lower guarantee needs to account for that failure event. This is standard robust dynamic programming, not a proven novel result.

## Reproduction and evidence boundary
Run `python research/six_issues_20261116/robust_horizon_crossing_20261117.py -v`. Local execution before push: 3 tests passed (200 one-step random instances; 80 horizon monotonicity/interval comparisons; invalid/zero cases). This is NOT a full scientific benchmark, and CI on the exact pushed SHA remains to be verified. The code's C is a caller-supplied predicate, not an anytime certificate implementation.

## Matched-information and matched-budget gate
Same feasible Q/H, base policy, KL constraint, initial (n,s), unknown Bernoulli preference environment, paired per-edge label streams, delta, confidence sequence, certificate predicate, total comparisons including warm-up, and after-every-label decision cadence. Each method receives only observed past labels; the robust planner's intervals must be computed from those same labels, not oracle p. Score paired crossing-by-budget, first valid certificate, false positives against truth, wall clock, and regret/native objectives separately. Compare with strongest Beta-mixture greedy, HCQA, robust Nash alignment, Nash learning, POSTPI, and adaptive preference learning (including strong multistep planners). No such direct full comparison has been executed here.

## Failure criteria and next exact step
Replace placeholder C with existing simultaneous beta-mixture certificate and confidence sequence; verify interval coverage under adaptive stopping; run paired benchmark against strong anytime greedy and HCQA; evaluate whether robust planning materially improves certified successes at equal query and compute budgets. If not, mark NO-GO. The Bellman/rectangular minimax theorem alone does not establish scientific novelty.

# Finite-horizon certificate-crossing DP — scoped result (2026-11-17)

## Algorithm
For a fixed candidate policy set Q, opponent set H, observed edge counts n and successes s, and budget B, freeze predictive Bernoulli parameters p_e=s_e/n_e at the beginning of planning. A state is (n',s',t). Define C(n',s') as the indicator that the current max-policy/min-opponent lower certificate exceeds threshold. The algorithm uses the same deterministic certificate as the previous one-step crossing code, with a configurable radius c/sqrt(n); this radius is **not** claimed anytime-valid.

V_0(z)=C(z). For t>0, V_t(z)=1 if C(z)=1, otherwise max_e[(1-p_e)V_{t-1}(z+e,success=0)+p_e V_{t-1}(z+e,success=1)]. Choose a maximizing edge. This is adaptive finite-horizon planning, not a fixed allocation.

## Theorem (conditional, limited)
Under independent edge-wise Bernoulli future labels with **known frozen probabilities** p_e, and a fixed deterministic crossing predicate C, V_B is the maximum probability of ever crossing by B additional comparisons over all adaptive edge-selection strategies. It weakly dominates the probability attained by the one-step crossing-greedy policy with the same p, C and B.

Proof: terminal and already-crossed states have values 0/1. Conditional on the first edge, the two Bernoulli outcomes partition the next state. Backward induction gives the optimal continuation for both outcomes; maximizing over the first edge yields the Bellman recursion. Greedy is a feasible strategy, so its value cannot exceed V_B. This theorem is standard finite-horizon dynamic programming, **not a globally novel theorem or a guarantee for the true unknown p**.

## Evidence
Code and tests: horizon_crossing_dp_20261117.py. Locally, 3 unit tests passed, including 120 seeded random instances comparing DP to a separately evaluated one-step greedy strategy. No measured improvement magnitude or statistically significant superiority is asserted. The exact committed file should also be run in CI before treating repository tests as verified.

## Fair comparator protocol
Hold fixed Q, H, baseline policy, KL cap, initial counts and successes, per-edge Bernoulli sample streams, confidence procedure, stopping criterion, total comparison budget B (including initialization), and decision opportunity after every label. Use the same frozen predictive model only for a **planning-model** comparison; separately assess true unknown preferences and coverage.

- Robust Nash Alignment: compare a robust Nash optimizer allowed the same uncertainty set and adaptive querying, evaluated also on the common crossing objective; native Nash objectives differ.
- Nash Learning from Human Feedback: allow preference cycles, same opponent access and query budget; evaluate both native game regret and the common crossing metric.
- POSTPI: allow valid post-selection confidence, same baseline and policy class; its high-confidence improvement assumptions may not map directly to preference games.
- APO/PILAF and active preference learning: allow adaptive pair choice and same data reuse, information and budget; a one-step strawman is insufficient.

## Open scientific gate
The Bellman theorem alone does not establish originality: exact belief-state or plug-in finite-horizon DP is a classical method. Scaling is exponential in B and the frozen plug-in model may be wrong. Required next: replace the plug-in model by a calibrated uncertainty-aware planner, preserve anytime-valid certificates, compare against strong adaptive baselines on paired data, and establish either a nontrivial separation or a NO-GO verdict. Do not call this independent novelty.

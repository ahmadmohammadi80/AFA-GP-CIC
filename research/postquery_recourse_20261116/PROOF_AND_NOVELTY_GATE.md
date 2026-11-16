# AFA — joint post-query policy recourse with anytime beta-mixture certificates

2026-11-16. **Independent worldwide novelty NOT established.** This is an executed research/algorithm artifact, not a manuscript.

## Problem and fair one-step theorem

Fix a baseline b, a finite list of KL-feasible candidate policies Q, a fixed set of opponent distributions H and a stationary Bernoulli pair-comparison kernel W (possibly cyclic). Confidence sets on W are simultaneously anytime valid, computed using Beta(1/2,1/2) likelihood-mixture e-processes with error delta/E per edge. Every adaptive policy selection, query and stopping rule then inherits the same certificate safety event.

Given observations D, a planned pair e and a fresh block X of B comparisons, let L_q(D,e,X) be the anytime lower confidence bound on the worst-opponent improvement of q after the hypothetical batch. The posterior predictive used to schedule queries is Beta-binomial(B, wins_e+1/2, losses_e+1/2), solely a planning device.

Exact one-block policy-recourse query value:
 V_joint(e) = E_X[max_{q in Q} L_q(D,e,X)].
Precommitted policy (but optimally chosen using the same information) value:
 V_precommit(e) = max_{q in Q} E_X[L_q(D,e,X)].

**Theorem:** max_e V_joint(e) >= max_e V_precommit(e). Proof: for each fixed q and each outcome X, max_j L_j(D,e,X) >= L_q(D,e,X). Integrate and maximize over q then e. Strict inequality is possible when the maximizing policy changes across outcomes of positive posterior probability. Exact positive witness: two-action b=(.5,.5), q1=(.7,.3), q2=(.3,.7), H={e0,e1}, edge with 15 wins out of 30 and B=20; expected policy-recourse bound exceeds optimal precommit-bound by >0.015.

The test fully grants the precommit comparator the optimal e, same posterior, same observations, and best precommitted q. Both sequential policies are allowed to switch q in future rounds; recourse optimizes the immediate one-block score. This is NOT a strict guarantee over unrestricted adaptive Nash or active preference learners, which can also optimize the recourse objective. Moreover, expectation under the Beta posterior is NOT a frequentist guarantee of reduced label complexity.

## Reproducible end-to-end test

Using identical synthetic stationary W, candidates, opponents, per-edge counterfactual Bernoulli streams, and 1,510 comparison budget:
  - "joint": exact one-block recourse edge selection with batch size 50;
  - "precommit": best exact fixed-q one-block edge selection, with re-selection possible at subsequent rounds;
  - "greedy_online": the stronger pre-existing anytime beta-mixture baseline, querying one edge per label rather than 50 at a time.

All three use the same per-edge labels and level-delta anytime martingale certificates. The greedy baseline is granted the fine-grained query clock; this **makes it a serious comparator**, although computational effort differs. The experiments are NOT real preference judgments or a global novelty proof.

**Novelty gate:** Earlier AFA already had a one-step policy-switching VOI variant using Hoeffding bounds (src/cert_voi/afa_cic_joint.py); this contribution upgrades that idea to the more powerful anytime Beta-mixture setting. Expected max / max expected recourse is standard stochastic-programming/value-of-information reasoning. Related theory: Carlsson et al. (2024), Pure Exploration in Bandits with Linear Constraints; Zhao and Jun (2026), Nearly Optimal Active Preference Learning; Nan et al. (2026), Efficient Exploration for Iterative Nash Preference Optimization; Das and Basu (2026), Learning to Explore With Lagrangians For Bandits Under Unknown Constraints.

For a publishable independent claim, the algorithm still needs a novel instance-dependent lower bound and a matching policy/query rule with a provable advantage against the best literature-level comparator. This artifact only advances implementation and falsifiable fair benchmarking.

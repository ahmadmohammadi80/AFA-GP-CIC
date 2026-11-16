# Active certificate budget oracle — exact result and limitations (2026-11-16)

## One algorithmic candidate: active-row budget oracle (ARBO)
Given a **single fixed active certificate row** with known weights w_e=|A_{q,h,e}|c_e, allocate a total B > 0 of comparisons across positive-weight edges to minimize the surrogate width F(n)=sum_e w_e/sqrt(n_e). The weights and radius model are known to the oracle; this is an offline target and a fair comparator for a future adaptive allocation algorithm, **not a validated online method**.

## Theorem (continuous optimum)
For w_e>0 and B>0, the unique minimizer of F(n) subject to n_e>0 and sum n_e=B is

n_e^* = B w_e^(2/3) / sum_j w_j^(2/3).

The optimal width is F^* = (sum_e w_e^(2/3))^(3/2) / sqrt(B).
Edges with w_e=0 need no comparisons for this fixed row.

**Proof.** Each term w_e n_e^(-1/2) is strictly convex for n_e>0. The Lagrangian stationarity condition -w_e/(2n_e^(3/2))+lambda=0 yields n_e=(w_e/(2lambda))^(2/3). Summing and enforcing the budget yields the expression. Strict convexity proves global uniqueness. Substitution gives the width formula. The zero-weight edge conclusion follows because its allocation cannot reduce the objective. QED.

## Integer optimum
For integer B and positive weights, DP[i,b] = min_{1<=n<=b-i+1} DP[i-1,b-n] + w_i/sqrt(n), with DP[0,0]=0, yields the exact integer optimum by optimal substructure; O(E B^2) time, O(E B) memory. This is an exact benchmark rather than a large-scale solver.

## Matched example (locally executed, Python unittest 3/3)
weights=[4,1,0,0.5], B=24, same row, same uncertainty model, same label budget. Equal active-edge allocation width 1.9445436482630054; continuous optimum 1.7255817569568541; integer optimum 1.7272932581135607 at [14,6,0,4]. This is a deterministic objective comparison, NOT stochastic preference learning performance, and not a claim against the published baselines.

## Scientific novelty gate
- Robust Nash Alignment and Nash Learning: game-theoretic policy objectives; a direct performance claim needs matching finite W, pi0, KL ball, pure/mixed opponent class, and identical queried labels.
- POSTPI: a policy improvement guarantee is not identical to a Nash equilibrium criterion. A shared positive certificate and error delta must be defined before comparing label counts.
- Active preference learning: nearest comparator for selecting informative pairs. It must be allowed the same coefficients, active row, uncertainty intervals and budget B; otherwise any observed gain would be a privileged-information artifact.
- MACWA optimizes a one-step maximum across rows. ARBO exactly optimizes a fixed single row over the remaining budget, but **does not solve the multi-row adaptive, unknown-W, anytime stopping problem**. Known convex allocation and 2/3-power rules may predate this construction; priority is not established.
- Actual user goal (meaningful distinguishable algorithm vs literature) remains OPEN. The present theorem is correct within its surrogate model but does not establish novelty or a better label-complexity bound.

Next: construct an anytime-valid plug-in/optimistic multi-row allocator with an explicit exploration floor, prove its adaptive bound, and benchmark against MACWA, greedy and literature-faithful active preference methods under matched information, budget, decision cadence, and common Bernoulli edge prefixes. Reject novelty if the gain disappears.

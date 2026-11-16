# AFA-GP-CIC — Monotone Confidence-Safe Elimination (MCE), research gate

## Single algorithmic candidate

Given fixed finite policies Q, opponent set H, a baseline b and an anytime confidence sequence for every queried edge, retain the intersection of each edge's successive confidence intervals. Define

L_t(q) = min_h sum_e min(A(q,h,e) l_{t,e}, A(q,h,e) u_{t,e})

U_t(q) = min_h sum_e max(A(q,h,e) l_{t,e}, A(q,h,e) u_{t,e}).

Set B_t = max_{s<=t, q in Q} L_s(q). Eliminate q permanently once U_t(q)<B_t. The next query can be selected exclusively from S_t = {e : exists surviving q,h with A(q,h,e) != 0}. Unlike static support pruning, this support can contract after observations. All original candidates remain in the mathematical upper bound used for the conservative simple-regret certificate.

## Theorem (correctness and monotonicity; exact arithmetic)

Assume all edge intervals contain their true stationary preference means at all times with joint probability >= 1-delta, including under adaptive edge sampling. Then simultaneously for all times:

1. Every maximizer of the true robust gain g(q)=min_h sum_e A(q,h,e)W_e remains alive.
2. The surviving set and S_t are nested and nonincreasing.
3. The regret of the policy qhat_t maximizing L_t among survivors is at most max_{q in Q} U_t(q)-L_t(qhat_t).
4. For every future certificate computation restricted to surviving policies, deleting all edges outside S_t is algebraically exact.

Proof: On the joint event, L_t(q)<=g(q)<=U_t(q) for every q and t. Every historical lower bound B_t is <= g* = max_q g(q). Thus for any optimal q*, U_t(q*)>=g*>=B_t, and q* is never eliminated. Permanent elimination makes surviving sets nested, hence their coefficient supports nested. Every removed edge has zero coefficient for every surviving policy/opponent pair; removing it changes no certificate. Finally g*<=max_q U_t(q) and g(qhat_t)>=L_t(qhat_t), proving the gap. This does NOT bound cumulative regret or establish an instance-optimal query complexity.

## Computational witness and tests

The executable witness in `monotone_elimination.py` has 3 policies, 4 possible edge coordinates, and after three valid updates retains one policy and one active edge. The test suite checks preservation of a true maximizer over 120 independently generated skew-preference vectors and 90 adaptive confidence updates per vector, as well as monotone support and fail-closed behavior on contradictory intervals. These are numerical checks, not a substitute for the proof.

## Novelty and matched-information gate

**Global independent novelty is NOT established.** Safe arm elimination and confidence-set intersections are classical in active learning and best-arm identification; this method may be a direct adaptation. The theorem is a scoped new statement about the specific robust KL policy-certificate operator, not a claim that its mathematical technique is globally new.

All competitors must see the same b, Q, H, KL radius, edge index and initial observations; each new preference comparison costs one label; the same fixed total label budget and stopping threshold are used. Our method does not get access to hidden true W or future labels. A static support-aware active preference baseline is a more meaningful comparator than uniform all-edge querying.

Robust Nash Alignment (arXiv:2610.00715) and Nash Learning (Munos et al., ICML 2024) target Nash-style game solutions, so a faithful matched-objective comparison must give them a common terminal safe-improvement certificate rather than pretending their original objectives match. POSTPI (Tse et al., RLC 2025) targets high-confidence policy improvement but its feedback and policy class must be aligned before comparison. Nearly Optimal Active Preference Learning (Zhao and Jun, 2026) is particularly close to dynamic elimination and must be treated as a potential prior-art overlap, not an automatic weak baseline.

## Remaining blocking test

A strong support-aware elimination baseline, run on exactly the same confidence process and edge streams, must be implemented and compared. Unless a distinct nontrivial sample-complexity theorem or statistically supported advantage remains, reject MCE as the central independent novelty. No manuscript or global novelty certification is authorized by this change.

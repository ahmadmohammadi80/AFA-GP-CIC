# AFA-GP-CIC: anytime power repair + explicit all-adaptive benchmark

Date: 2026-11-16. This is a code/proof audit, NOT a manuscript. Global independent novelty is not yet proven.

## Why this stage exists

The earlier edgewise time-uniform union-Hoeffding policy selector yielded **0/40** certificates under a budget of 1,510 pairwise labels. The current stage tests a valid, usually tighter alternative, not a loose invalid fixed-time certificate.

## Theorem 1: reusable beta-mixture policy certificate

For each unordered edge e=(i,j) with stationary independent Bernoulli X_(e,t) of unknown probability p_e in [0,1], define s_e as the number of wins in n_e observations. Consider a Beta(1/2,1/2) mixture over hypothetical alternatives theta. Against null p_e, its likelihood ratio is

 M_(e,n)(p) = Beta(s_e+1/2,n-s_e+1/2) /
               [Beta(1/2,1/2) p^s_e (1-p)^(n-s_e)].

Under the true p_e, M_(e,n)(p_e) is a nonnegative mean-one martingale, being a mixture of likelihood-ratio martingales. By Ville's inequality,
 P_p(there exists n: M_(e,n)(p_e) >= E/delta) <= delta/E.

Use alpha=delta/E and let [lo_e,hi_e] be the inverse interval of means 2p_e-1 for which log M_(e,n)(p) < log(E/delta). For n=0 the vacuous interval is [-1,1]. Intersections with [0,1] ensure finite-domain validity. With independent streams per edge, and an adaptive choice of which stream to observe next, the event that every stream's true p_e lies in its interval for every n has probability at least 1-delta. This event is uniform over ALL policies, opponent distributions and stopping times selected after seeing the data, since the event controls the whole W entrywise.

For reference policy b, candidate q and opponent h, define A_(q,h,(i,j))=(q_i-b_i)h_j-(q_j-b_j)h_i. Then:

 (q-b)^T W h = sum_(i<j) A_(q,h,e) * (2p_e-1).

For any post-selected q (including KL-constrained candidates), and any post-selected finite H, the lower certificate

 L(q,H)=min_(h in H) sum_e A_(q,h,e) * [lo_e if A_(q,h,e)>=0 else hi_e]

satisfies L(q,H) <= min_(h in H) (q-b)^T W h simultaneously with probability at least 1-delta.

Assumptions: fixed stationary Bernoulli probability per edge, conditionally independent observations in each edge's pre-generated sequence, known action labels, fixed baseline b, and a consistent known mapping between q, h, and W. This does not cover unrestricted nonstationary human feedback, label dependence, adversarial drift or unknown preference model misspecification.

**No originality claim for the beta-mixture confidence sequence.** The novelty target is the exact coupling with KL-constrained multi-opponent improvement under cyclic preference, which remains to be compared with the strongest published algorithms.

## Theorem 2: honest information limit for a known two-world subproblem

Use the two explicit legal cyclic matrices W+ and W- already proved in
research/cycle_sensitive_20261116/PROOF_AND_NOVELTY_FA.md.
The two worlds have identical Wb but differ in multi-opponent certificate feasibility.

For each edge e set p^+_e=(1+W^+_e)/2 and p^-_e=(1+W^-_e)/2. Any sequential, possibly randomized, adaptive edge-selection method that distinguishes the two worlds with error at most delta in each world and finite expected stopping time must obey, by likelihood-ratio chain rule + data processing,

 E_+[T] >= kl(1-delta,delta) / max_e kl(p^+_e,p^-_e),
 E_-[T] >= kl(1-delta,delta) / max_e kl(p^-_e,p^+_e).

For delta=.05, the universal lower bounds are 6.938822 and 7.478818 observations respectively, **including all adaptive edge-selection strategies** for this exact known-pair model.

Our implementation uses the fixed edge (0,1), with p+=.75 and p-=.35, and Wald sequential likelihood ratio stopping at +/- A, A=log(1/delta).

- Its error in either world is <=delta by the likelihood-ratio martingale and Ville.
- Write D+=kl(.75,.35), D-=kl(.35,.75), u=log(.75/.35), v=-log(.25/.65).
- By Wald's stopped log-likelihood identity, positive drift under plus and negative under minus, and bounded overshoot, its expected samples satisfy
  E_+[T] <= (A+u)/D+ and E_-[T] <= (A+v)/D-.
  The increments are bounded, drifts nonzero, and T is integrable.
- With delta=.05, these upper bounds are 11.294156 and 11.151202. Dividing by the above universal lower bounds gives **competitive factor upper bounds 1.627677 and 1.491039** with respect to the infimum over ALL adaptive methods for this EXACT two-world problem.

This is a proved near-information-limit claim in a deliberately narrow benchmark, **not** a general AFA sample-complexity theorem for unknown arbitrary W. Wald/likelihood-ratio testing and change-of-measure bounds are established sequential-analysis techniques.

## Empirical comparison (deterministic seeds, runnable CI)

Synthetic stationary preference game with K=5, two pre-specified KL-feasible q candidates, 5 pure opponents, nonzero 3-cycle, delta=.05, acceptance threshold .015 and a hard cap of 1,510 comparisons **including initial observations**:

- Prior valid anytime Hoeffding bound, same edge-selection rule: 0/40 positive certificates.
- Beta-mixture anytime bound: 34/40 positive certificates.
- No false positive certificates observed; this is empirical and does not prove zero risk.
- One of the 40 beta-mixture runs had an edgewise coverage violation; this is compatible with the valid delta=.05 confidence guarantee. Reporting it is essential.
- Null control W=0: 0/80 positive certificates for both methods, under a separate reduced-budget check.
- Per-edge randomness is coupled across methods: both see the same prefix for any given edge, even when the adaptive query schedules differ.

For the known W+/W- sequential subproblem, 10,000 trials per world yielded:
- minus world: 8.8658 mean labels, 369/10,000 classification errors;
- plus world: 9.1377 mean labels, 345/10,000 classification errors.

These trials are synthetic and offer **no** evidence of human preference alignment or general superiority to Robust Nash. Both methods used equal comparison budgets/assumptions in the power comparison; the mixture inversion is computationally costlier per queried edge.

## Final originality gate — still OPEN

Existing related work includes:
- Ahmed et al., Robust Nash Alignment under Preference Uncertainty (2026), https://arxiv.org/abs/2610.00715
- Zhao and Jun, Nearly Optimal Active Preference Learning and Its Application to LLM Alignment (2026), https://arxiv.org/abs/2602.01581
- Saad, Thuot and Verzelen, The Sampling Complexity of Condorcet Winner Identification in Dueling Bandits (2026), https://arxiv.org/abs/2603.15189
- Garivier and Kaufmann, Optimal Best Arm Identification with Fixed Confidence (2016), https://proceedings.mlr.press/v49/garivier16a.html

Stronger anytime certificates and classical sequential tests are already known. A strongest unrestricted adaptive method can simulate our method exactly, so a **strict universal advantage over ALL equally informed adaptive methods is logically impossible**. A defensible originality claim needs a precisely scoped published competitor, an actual algorithmic difference not present in that prior work, and a rigorous rate/constant/computation advantage with equal information and permitted decisions. This stage repairs **statistical power** and proves a **limited information competitive ratio**; it does **NOT** settle global independent novelty.


## Additional direct prior-art gate (verified 2026-11-16)

The overlap is stronger than just Robust Nash:
- Ramdas, Grünwald, Vovk and Shafer, "Game-Theoretic Statistics and Safe Anytime-Valid Inference" (2023/2024), https://www.imstat.org/publications/sts/sts_38_4/sts_38_4.pdf explicitly covers e-processes, test martingales and optional-stopping-safe confidence sequences.
- Jun, "Confidence sequences via online learning with applications in offline contextual bandits and active preference learning" (2025 JSM), https://ww2.amstat.org/meetings/jsm/2025/onlineprogram/abstract.cfm?sid=2118&tid=2120 ties modern tight confidence sequences directly to active preference learning.
- Nan et al., "Efficient Exploration for Iterative Nash Preference Optimization" (2026), https://arxiv.org/abs/2606.01382 already gives KL-sensitive exploratory Nash preference algorithms.

Therefore even a **valid 34/40 vs 0/40 improvement** over the former Hoeffding baseline does not establish an algorithmic improvement over the best-published anytime-valid confidence sequence, ENPO, active preference optimization, or Robust Nash. The theorem of within-1.63 cost for the two-known-world benchmark is substantive and fully scoped, but the use of SPRT and change of measure is classical. Independent world-first novelty is still UNCONFIRMED and should not be automatically scored as complete. Next comparison must let strong published methods adapt edge choices and use tight confidence sequences.

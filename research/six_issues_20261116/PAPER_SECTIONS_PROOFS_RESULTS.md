# AFA-GP-CIC: audited theorem, methods and experiment sections

Version: 2026-11-16. The claims below are scoped to specified finite stationary or explicitly modeled nonstationary preferences. This document is a draft research section, NOT evidence that originality has been independently certified.

## 1. Problem

Let K actions, E=K(K-1)/2 unordered queryable pairs, a full-support reference b and a known finite candidate set Q of distributions such that KL(q||b)<=kappa for all q in Q. Let H be a finite opponent set. No scalar reward representation is assumed, and W can have genuine sign cycles. For W_ij=-W_ji=2 Pr(i preferred to j)-1 define

  g_W(q) = min_{h in H} (q-b)^T W h.

An accepted policy must satisfy g_W(q)>tau with confidence 1-delta. For e=(i,j), a_(q,h,e)=(q_i-b_i)h_j-(q_j-b_j)h_i, hence each payoff is exactly sum_e a_e W_ij.

## 2. Deployed Beta-mixture anytime certificate

For an edge with s Bernoulli wins in n comparisons, set

  M_(e,n)(p)=B(s+1/2,n-s+1/2)/(B(1/2,1/2)*p^s*(1-p)^(n-s)).

Under a FIXED stationary pair mean p, M is a nonnegative test martingale. Ville's inequality with alpha=delta/E yields simultaneous coverage of p_e for every edge and every sample count; adaptive edge selection, post-selection of q and arbitrary stopping do not invalidate the coverage event. The minimum of the linear payoff over these edge sets is an anytime lower certificate.

This concentration method is established prior art, not a standalone new theorem.

## 3. Root-cause diagnosis and corrected joint scheduler

The initial one-block lookahead optimized expected numerical lower confidence after 50 labels, not the probability that the confidence lower bound would cross tau before a fixed label budget. Equalizing both algorithms to 50-label blocks did not repair the failure (joint 13/40, Beta-mixture greedy 36/40).

The corrected rule decides after EACH label. For a candidate q, choose its currently worst lower-confidence opponent h*(q). Define

  s_e(q) = |a_(q,h*(q),e)| * (u_e-l_e)/(sqrt(n_e)+0.2).

For q_top maximizing current lower certificate, select the next edge maximizing

  0.70*s_e(q_top)+0.30*max_{q in Q}s_e(q).

This is a fixed heuristic to protect the current information bottleneck while keeping potentially superior alternative policies active. It is not proven optimal, and 0.70 was fixed before the independent 200-seed evaluation.

**Verified primary results (all policies share the same 1510-label cap, KL cap, E-edge oracle streams and stopping opportunities):**

| Regime / runs | Greedy positive | Corrected joint positive | Statistical note |
|---|---:|---:|---|
| Initial 40 | 34/40 | 38/40 | McNemar exact p=.125; Holm p=.18457 |
| Second independent 40 | 34/40 | 37/40 | exact p=.375; Holm p=.375 |
| Genuine 2>3>4>2 sign-cycle, 40 | 35/40 | 40/40 | exact p=.0625; Holm p=.0625 |
| Additional predeclared independent 200, noncyclic | 143/200 | 182/200 | exact matched p=7.5972e-9; Holm for paired success and cost p=7.5972e-9 |

There are zero observed false-positive certificates in these runs, not a zero-risk guarantee. On the independent 200, the 39-certificate gain came with 7.385 MORE mean labels (1100.715 versus 1093.33). The exact paired two-sided label sign test with Holm is significant in the adverse direction (adjusted p=4.087e-13): joint used more labels in 143 matched trials, fewer in 44. Thus certification power improves, but label cost does not dominate.

The original synthetic preference matrix had nonzero three-edge curl but a transitive SIGN ordering; the additional test sets W_24=-.18, W_42=.18, making a true 2>3>4>2 sign cycle.

## 4. Published baseline protocols

RUCB (Zoghi et al., ICML 2014; https://proceedings.mlr.press/v32/zoghi14.html) uses optimistic pairwise preference bounds to choose a prospective champion and challenger. D-TS (Wu and Liu, NeurIPS 2016; https://papers.nips.cc/paper/6157) uses Beta posterior samples for Copeland/challenger selection. Both are reproduced as query selectors; as required for the new goal, they are evaluated with THE SAME final beta-mixture KL-gain certificate as AFA. Zero-information self-pairs are replaced by informative distinct-arm queries, an explicitly documented departure from the paper's full protocol.

Seeger's PAC-Bayes-kl result (JMLR 2002, https://www.jmlr.org/papers/v3/seeger02a.html) is used in a conservative point-mass specialization for the finite family of (q,h) hypotheses. For IID uniform edges e after a shared fixed pilot, set pi_e=1/E, B=max_(q,h,e)|a_(q,h,e)|/pi_e, and Z=(a_e Y_e/pi_e+B)/(2B) in [0,1]. Then E[Z]=( (q-b)^T W h+B)/(2B). For each n let delta_n=6delta/(pi^2 n^2). The iid PAC-Bayes-kl bound and Pinsker yield simultaneously for all finite hypotheses and all n:

  E[Z] >= mean(Z) - sqrt( log(2 sqrt(n)*|Q||H|/delta_n)/(2n) ).

This supports adaptive stopping and postselection over the FIXED finite class, but the reference edge sampling must remain IID after the pilot; it is not a PAC-Bayes validity claim for the adaptive RUCB/AFA edge schedules.

Under the original 40-seed capped certification comparison, both RUCB and D-TS with the common certificate obtained 0/40; the singleton PAC-Bayes method also obtained 0/40. These methods were published for different optimization objectives (Copeland/Condorcet regret or generalization), so their poor performance here does NOT establish AFA's global novelty or superiority at their original objectives. Genuine safe preference-policy baselines targeted at the exact KL-multi-opponent objective are still needed.

## 5. Proposition: a genuine cyclic two-world advantage over PRECOMMITTED policies

Set b=(1/5,...,1/5), epsilon=.08, q+=b+epsilon(e0-e1), q-=b-epsilon(e0-e1), kappa=.05. Let W+ be skew-symmetric with W+_01=.6; W+_0j=.6, W+_1j=-.4 for j in {2,3,4}; W+_23=.3, W+_34=.3, W+_42=.3. Set W-=-W+. Each world has a sign cycle.

For pure opponents, g_(W+)(q+)>=.048, g_(W-)(q-)>=.048, and the reversed choice fails. Under a uniform prior on the two known worlds, a policy precommitted before feedback can choose the correct q with probability at most 1/2. However, querying edge (0,1) n times and choosing q according to the observed win majority has error at most exp(-.18n) in each world by Hoeffding since p+=.8 and p-=.2. At delta=.05, n=17 suffices, and the correctly selected q has worst-opponent improvement at least .048. The finite two-world synthetic test achieved 40/40 correct choices in W+ and 39/40 in W-.

Important restriction: competitors are forced to precommit q before data, and the two worlds are known in advance. An unrestricted adaptive competitor can emulate the joint algorithm; no novel general information-theoretic separation is established by this proposition.

## 6. Theorem: high-probability simple regret and explicit sample cap for GUARDED joint selection

The GUARDED variant uses the same joint candidate-selection score but forces a round-robin edge query every m-th post-pilot label. Its confidence set is the intersection of Beta-mixture and standard all-n Hoeffding intervals with error probabilities split delta/2 and delta/2.

Let S=max_(q,h) sum_e |a_(q,h,e)| and

  r(n)=sqrt(2 log(2E pi^2 n^2/(3delta))/n).

For every T, let n_min(T) be the smallest edge count and qhat_T be the policy maximizing the simultaneous lower bound. **Theorem:** with probability >=1-delta, simultaneously for ALL T,

  0 <= g(q*)-g(qhat_T) <= 2 S r(n_min(T)),

where q* is a best candidate in Q. Proof: on simultaneous interval coverage, 0<=g(q)-L_T(q)<=max_h sum_e |a_e|(hi_e-lo_e)<=2Sr(n_min). Using L_T(qhat)>=L_T(q*) and g(qhat)>=L_T(qhat), the result follows. This argument is valid for arbitrary adaptive edge selection as long as each interval is valid.

Since |g(q)|<=2, its expected simple regret is bounded by 2Sr(n_min)+4delta using the deterministic lower bound on n_min supplied by forced exploration. If the true positive margin Delta=g(q*)-tau>0, choose n_star>=10 such that 2Sr(n_star)<Delta. Every edge has >=n_star queries by

  T_star = E + m E (n_star-1).

On the same coverage event a positive certificate is then issued by T_star. On the original 40-seed example, the GUARDED variant certified 36/40, with no observed false certificates; a conservative exact numerical sample cap was 511,810 labels, which is too large for a practical optimality claim.

**General information limit:** for two specified different Bernoulli preference worlds W,W' with disjoint delta-correct answers, the KL change-of-measure and data-processing inequalities give

  E_W[T] >= kl(1-delta,delta) / max_e kl(p_e^W,p_e^W').

This is the universal binary-distinguishability lower bound. It does not imply a universal constant-factor comparison with T_star. The factor 1.63 previously reported belongs only to another KNOWN two-world SPRT example; transferring it to general unknown W would be incorrect.

## 7. Theorem: current-time certificate for gradual bounded drift

The stationary Beta-mixture martingale is not valid under unrestricted drift. Instead suppose each pair has predictable conditional win probability p_(e,t) and a KNOWN bound |p_(e,t)-p_(e,t-1)|<=rho per calendar time, with independent bounded feedback conditional on past history. If edge e was sampled at calendar times tau_1,...,tau_n, let m_e be the mean of signed observations and bar_tau the mean query calendar time. Define

  r_e(n)=sqrt(2 log(E pi^2 n^2/(3delta))/n),
  bias_e(t)=min(2,2rho(t-bar_tau)).

Then with probability >=1-delta, simultaneously over all pairs, sample counts and calendar t,

  mu_(e,t) in [m_e-r_e(n)-bias_e(t), m_e+r_e(n)+bias_e(t)] intersect [-1,1].

At n=0 the interval is [-1,1]. Proof: Hoeffding-Azuma for the predictable selected-edge martingale differences gives the same 6delta/(E pi^2 n^2) tail bound; sum over all n,e. The deterministic drift assumption bounds the difference between the average past conditional mean and the current mean by 2rho(t-bar_tau); apply triangle inequality. This yields a valid time-t certificate for all adaptive policies using the coefficient identity.

An adaptive-sampling synthetic test with 40 independent paths, 500 time points each, found 40/40 covered over all edges and calendar times; exact 95% lower confidence limit for observed path coverage was .9119. However mean width 1.898 is broad, limiting statistical power. Thus this is a valid but conservative dynamic extension, not evidence of dynamic regret optimality.

## 8. Known-change-point alternative

When a finite set of J change times is PUBLIC and within each epoch all edge Bernoulli means are fixed, restart the Beta-mixture martingale at each known boundary and use delta/(J E) per edge per epoch. Ville plus union over epochs and edges gives 1-delta simultaneous coverage of W_t under arbitrary adaptive queries, stopping times, and policy selection in every epoch. This covers abrupt nonstationarity with known change points. It does not justify using this reset rule if change points are learned from the same data without accounting for selection. A 40-path, 3-regime, 1510-label-per-regime experiment is included in CI.

## 9. Exact computational optimization

For pure opponents H={e_j}, each edge (i,j) contributes only to the i and j opponent coordinates. A dense routine calculating Q*K*E coefficient entries requires O(Q K^3) work because E=Theta(K^2); the exact specialized routine calculates the contributions using O(Q E)=O(Q K^2) operations and O(Q K) auxiliary score storage.

The fast and dense certificate values differed by at most 1.7e-16 in tests up to K=80. Measured speedups on one GitHub runner at K=5,10,20,40,80 were 2.66x,3.18x,4.99x,10.21x,20.71x respectively. Peak memory reductions measured by Python tracemalloc are not complete process-RSS measurements. A scaling plot and JSON audit are uploaded as CI artifacts.

## 10. Research acceptance status

1. Issue 1, statistical power: repaired in the specific held-out 200-seed original stationary family, with corrected two-sided multiplicity-adjusted p=7.5972e-9; no label-cost superiority and independent originality not proved.
2. Issue 2, published baselines: RUCB, D-TS and a PAC-Bayes-kl singleton specialization implemented and run. Published objective and safe-KL objective still differ. Not sufficient as state-of-the-art matched-objective coverage.
3. Issue 3, theoretical joint selection: constructive proof vs the precommitment-restricted class. Not a theorem beating unrestricted adaptive methods.
4. Issue 4, regret: all-time simple regret and finite certificate sample cap for the explicitly GUARDED algorithm. Minimax matching and guarantees for the unguarded high-power algorithm remain open.
5. Issue 5, dynamic preferences: two rigorous restricted extensions (known bounded slope; known regime boundaries), with simulations. Unknown changes or nonparametric nonstationarity remain open.
6. Issue 6, scaling: exact pure-opponent computational improvement with reproducible time/memory plots. Not a complete end-to-end profiling across general H and GPU models.

No missing theorem, label-cost result, or literature comparator is to be silently filled with a fabricated success claim.

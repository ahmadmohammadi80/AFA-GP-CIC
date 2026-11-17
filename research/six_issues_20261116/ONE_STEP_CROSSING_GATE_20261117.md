# One-step certificate-crossing query oracle — scoped theorem and novelty gate (2026-11-17)

**Status:** Candidate decision rule, not a demonstrated novel scientific contribution. No empirical superiority over strong active baselines has been established.

## Exact rule
Let each edge e have n_e independent Bernoulli preference labels with s_e successes. For each candidate policy q and opponent h, let a_{qhe} be the fixed linear coefficients of its preference gain, and let r_e(n) be a supplied confidence radius. At the current state, define the observed lower certificate
L(s,n) = max_q min_h [sum_e a_{qhe}(2s_e/n_e-1) - sum_e |a_{qhe}| r_e(n_e)] - tau.
For each available next edge e, enumerate the two possible next labels y in {0,1}, update only n_e and s_e, and compute I_e(y)=1{L(s+y*unit_e,n+unit_e)>0}.
Define p_hat_e=s_e/n_e and
P_hat_e=(1-p_hat_e)I_e(0)+p_hat_e I_e(1).
Choose argmax_e P_hat_e, breaking ties by lowest edge index.

## Theorem (exact one-step plug-in objective)
For fixed s,n, coefficient tensor, threshold tau, and radius function, the algorithm returns an edge maximizing the conditional **plug-in** probability of a positive certificate after exactly one additional Bernoulli comparison. Runtime O(E^2 QH) with direct recomputation, memory O(E+QHE) counting inputs.

**Proof:** For any selected edge e the next Bernoulli label has exactly two possible values, with plug-in masses 1-p_hat_e and p_hat_e. The corresponding updated certificate values are computed explicitly. Their weighted indicators sum to the exact probability under the plug-in model. Enumerating all eligible edges and taking the maximum therefore returns a global optimum of this one-step objective. QED.

**Crucial limitations:** This is NOT an original concentration result, a nontrivial finite-horizon regret/sample-complexity bound, or a guarantee on the true probability of certification: p_hat_e is an estimator. A valid anytime certificate requires the externally supplied radius function to be simultaneous and time-uniform. Adaptive policy choice is safe only under that preexisting simultaneous-coverage event. The implementation is a planning oracle, not a confidence construction. The theorem may be a standard two-outcome lookahead and does not itself establish independent novelty.

## Reproduction
Run: `python research/six_issues_20261116/one_step_crossing_20261117.py -v`.
The file contains two tests: 300 randomized small cases independently enumerating both outcomes for every edge, and malformed-input checks. A related local prototype passed 3 tests (including 300 random cases) before push; **the exact committed file's CI result has not been verified**.

## Matched-information and matched-budget comparator gate
All methods must have the same b, KL cap, feasible Q, H, initial (n,s), same preference matrix, same per-edge Bernoulli streams, same total query cap including initialization, same decision cadence, same time-uniform certificate and delta. At each step all methods see only previously sampled outcomes. Evaluation: fraction of runs with valid positive certificate by budget, paired first certification time, false certification against ground truth, and CPU time. Compare against the strongest Beta-mixture greedy, HCQA, active learning query design, and suitably adapted Robust Nash Alignment / Nash Learning / POSTPI. Report native objectives separately; do not treat non-equivalent goals as interchangeable. No such comparative experiment is claimed here.

**Next falsification gate:** compare this exact one-step oracle with strong greedy under matched streams and budget; if the oracle fails, stop claiming a meaningful independent algorithmic advance and investigate multi-step planning rather than renaming the method.

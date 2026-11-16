# Matched-information, matched-budget novelty gate

The tested new component is **anytime, policy-reusable preference certificates**. This is a valid technical extension, but the time-uniform confidence bound and post-hoc uniformity are known principles; independent global originality remains **unverified**.

## Fair comparison protocol

All algorithms must receive the same action set, baseline policy b, KL cap, preference-query oracle with the same Bernoulli outcomes/distribution, candidate-policy access, opponent access, initial information and total comparison budget including exploration. A method may adaptively choose which edge to query, when to stop and which q to report. All report the same worst-opponent gain criterion and the same false-certification tolerance delta. Distinguish oracle labels, CPU/GPU cost, and number of comparisons. For post-hoc policy selection, either every method uses a simultaneous/time-uniform certificate or its own valid adaptive test; a fixed-time bound applied after optional stopping is not an acceptable baseline.

## Relevant prior work and why it matters

- [Robust Nash Alignment under Preference Uncertainty](https://arxiv.org/abs/2610.00715): uncertainty-aware Nash preference games. A robust Nash baseline must be allowed to use the same confidence set, not a weaker point-estimate payoff matrix.
- [Nash Learning from Human Feedback](https://proceedings.mlr.press/v235/munos24a.html): preference games with cyclic/intransitive feedback. Cycling alone is not novel.
- [POSTPI](https://rlj.org/papers/tse2025high): high-confidence policy improvement. Allow a POSTPI-like selector to reuse data with an anytime-valid bound and optimized query allocation where its assumptions permit.
- [Active Preference Optimization (APO)](https://arxiv.org/abs/2402.10500) and [PILAF](https://proceedings.mlr.press/v267/feng25g.html): active preference sampling. Compare to an active edge selector, not only uniform querying.

## What is and is not proven

The edgewise event holds simultaneously over all policies and opponent sets for stationary independent per-edge outcomes; this permits choosing a policy after observing labels and stopping adaptively without a held-out confirmation sample. This is a correctness theorem, **not** an information-theoretic separation. The next gate is an explicit instance family and matching lower bound that survives the strongest fair baseline above, or an honest NO-GO on the independence claim.

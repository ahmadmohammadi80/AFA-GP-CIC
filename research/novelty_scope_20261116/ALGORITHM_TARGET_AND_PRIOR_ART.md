# AFA-GP-CIC — novelty gate after numerical/theoretical closure

Date: 2026-11-16. **Algorithm research note, not a manuscript.**

The fixed-baseline finite-action solver is now implementation/validation complete in
the scope documented at `research/verified_closure_20261116`.
The independent global novelty claim is **NOT yet verified**.

## Closest prior work verified

| Work | Established contribution | Consequence for AFA claims |
|---|---|---|
| Jiang et al., *HodgeRank*, Math. Programming 2011 | Gradient/cyclic decomposition of pairwise rankings | The mere discovery of divergence-free cycles and Borda score loss is **not new** |
| Munos et al., *Nash Learning from Human Feedback*, ICML 2024 | Nash optimization on pairwise preference kernels, including cyclic comparisons | Optimizing cyclic preferences without scalar reward is **not new** |
| Tse et al., *POSTPI*, RLC 2025 | High-confidence improvement from preference feedback | A generic preference safety certificate is **not new** |
| Ahmed et al., *Robust Nash Alignment under Preference Uncertainty*, arXiv:2610.00715, 2026 | Four-player robust Nash game with preference-kernel ambiguity, convergence and LLM experiments | KL robust preference-game optimization is **not new** |
| Zhao & Jun, *Nearly Optimal Active Preference Learning*, arXiv:2602.01581, 2026 | Instance-dependent active preference-label complexity and a greedy scheme | Generic adaptive preference queries / sample-efficiency claims are **not new** |

Sources:
- https://link.springer.com/article/10.1007/s10107-010-0419-x
- https://proceedings.mlr.press/v235/munos24a.html
- https://rlj.org/papers/tse2025high
- https://arxiv.org/abs/2610.00715
- https://arxiv.org/abs/2602.01581

## Concrete next algorithmic object (proposed, **not proven novel**)

**Cycle-sensitive active multi-opponent certification**, not simple
KL full-simplex optimization. Fixed baseline `b`, finite challenger family
`H`, unknown skew kernel `W`, fixed KL radius `kappa` and target
gain `gamma`. For pairwise edge confidence region `U_t`:

```
V_t = max_{q ∈ simplex, KL(q||b)≤kappa}
          min_{h∈H, W'∈U_t} (q-b)^T W' h
if a feasible q has numerically audited lower margin > gamma: emit q
otherwise: query an edge expected to shrink the active worst-opponent
          bottleneck under the same total comparison budget
```

**What is actually needed to establish independence:** a new query-selection
operator with a theorem showing finite-sample / computational separation
against a strong, same-information method that also uses the entire skew
kernel and multiple opponents. Compare against best-adaptive tracking and
Robust Nash (not against a single-transfer-only or Borda-only strawman).
The 2026 two-world construction establishes loss under compression to
`Wb`, but **does not** establish any advantage over unrestricted edge-aware
adaptive learners.

**Proof obligations still open:** exact joint edge-uncertainty treatment
rather than independent-score rectangles, optimal query rule, lower bound
under the same information state, independent baseline implementation,
and feedback from human preferences. No numerical results for this proposed
operator are claimed here.

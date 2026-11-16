# Reproduced 40-trial audit — 16 November 2026

GitHub Actions evidence from the scaled optimizer: https://github.com/ahmadmohammadi80/AFA-GP-CIC/actions/runs/43859410698

Fixed KL-feasible candidate q, ten actions, six fixed opponent policies, fourteen relevant pairwise edges, delta=0.05, threshold=0.02, independent 100-label pilot per relevant edge. Synthetic stationary pair preferences with a genuine three-cycle.

- 8/8 unit tests passed.
- 40/40 numerical optimization calls reported success.
- 40/40 independent confirmation trials produced positive certificates for the proposed allocation.
- 40/40 independent confirmation trials also produced positive certificates for the informed, uniform-over-relevant-edges comparator at the *same confirmation label budget*.
- Mean proposed confirmation labels per trial: 18,737.475.
- Mean number of uniform-over-relevant-edges labels needed to satisfy **the same pilot-derived per-opponent width constraints**: 41,634.6.
- Relative reduction in the required *design width target* comparison budget: about 54.996%.
- Mean largest unnormalized confidence-width contribution: 0.02078 for proposed design versus 0.01696 for uniform at the *same* label budget. The unnormalized-width metric favors uniform; target constraints are different for different opponents.

These are **different evaluation questions**: feasible design cost under a prescribed width condition versus realized certification at a fixed equal budget. The proposed method improved the former on this experiment but **did not outperform** the comparator on certification frequency. Pilot labels are additional cost and are identical for both arms; they are not omitted from a hypothetical total-effort claim.

Uncertainty notes: the metric is specific to finite stationary preference games and a given fixed q/H, and the multipliers dual bound is not yet tight for every multiple-opponent instance. Current experiments do not establish rate advantages over strong unrestricted adaptive active preference learning. This is a candidate mechanism and a testable lemma, not an independent global novelty claim.

Prior art: Ahmed et al. 2026 robust Nash alignment, Nan et al. 2026 exploratory Nash, Carlsson et al. 2024 linear constrained pure exploration, Garivier & Kaufmann 2016 tracking.

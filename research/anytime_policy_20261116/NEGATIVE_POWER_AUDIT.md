# Empirical smoke audit — 16 November 2026

Source: https://github.com/ahmadmohammadi80/AFA-GP-CIC/actions/runs/43869075507
Artifact: anytime-policy-smoke (raw 40 rows).

- 6/6 Python unit tests pass.
- 40 seeded sequential simulations executed with five actions, two candidate policies, a KL cap of 0.05, delta=0.05, and at most 1,500 additional edge comparisons after ten initialization comparisons.
- **0/40 certificates issued** within that budget; **0 false certificates** (vacuously).
- Average labels per trial: 1,510; all trials reached the cap.
- A typical true worst-opponent gain was 0.0256 against the certification threshold 0.015, yet the conservative anytime lower bound stayed negative.

Interpretation: This is a **negative power result** at the tested budget, not proof of usefulness or superiority. The confidence sequence gives a valid post-selection safety theorem but is too conservative under the current simple edge rule and budget. We do not assert novelty or empirical success. To advance, compare a time-uniform optimized-allocation baseline and improve sample complexity without weakening validity; do not compare only with an invalid fixed-time optional-stopping baseline.

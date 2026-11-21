# APA independent human replay — EXECUTION CODE LOCK
Date 2026-11-21. This lock was created **before reading any heldout ballot preference outcomes or computing their pairwise kernels**.

Previously committed protocol anchor: research/independent_apa_20261121/PREREGISTRATION_LOCK.md.
Canonical protocol JSON SHA256: 22e6266dce3332a7379808b44c6dbcbb701bef1bdfb8d4b95753dac60d40c940.

Executable local artifacts to be kept with that exact digest:
- apa_online_engine.py SHA256: 640cf613da0a82d4f6f30201c0032cf14deef1643b9ad4d89d5a7faf4cd6e380
- reference_math_source_aligned.py SHA256: 159361563b75c9fd5b64c1c538ae9897e7d86cb9eafeec2f7893c9b02e02df09
- apa_preflight.py SHA256: e42cb872ddb4be706dea566bab054e9ad01d157eb08266cfd787de313af0c582
- test_apa_preflight.py SHA256: 13b3cf5efdd49cdb81780ce42c536d71339222d7e870081cb4a6df0cc3599229

Software checks completed before this lock: 7/7 preflight unit tests and 20 synthetic training-kernel smoke trajectories passed; 0 certification on those 200-label *smoke* examples is not a human heldout result.

Code uses the source-aligned reconstruction of AFA, NOT an authenticated repository checkout. The label-generation design draws IID per-edge Bernoulli labels from the fixed heldout empirical kernel, which is distributionally equivalent to fresh heldout-respondent draws *with replacement* for every comparison. No individual ballots are published.

Frozen experiment: all methods AFA/Greedy/Joint-Mix/Round-Robin/Width; cap 8000; 40 full online seeds per election; first near-gate eligible state in 1200 prefix attempts for each of six origin-year strata, 24 suffix replicates, primary horizon 5, six-stratum paired cluster inference. SUSHI results remain historical development evidence.

A later change of any locked source byte or analytical choice requires a new publicly disclosed version. Neither the hashes nor code-lock demonstrate success or validate the original GitHub source equivalence; that needs a clean parity/CI run.

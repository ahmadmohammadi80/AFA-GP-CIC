# AFA-GP-CIC / CA-VOI

**Private research prototype**: certificate-aware active preference learning for finite-response alignment under possibly cyclic preferences.

The algorithm couples (i) KL-constrained policy-candidate selection, (ii) active pairwise comparison selection based on certificate-aware value of information, and (iii) anytime-valid acceptance/rejection of candidates. It does not assume Bradley–Terry or scalar rewards.

**Scope and caveats:** Results are synthetic. Worldwide scientific novelty, superiority to all published active-learning methods, and generalization to large language models are **not yet verified**. The probabilistic guarantees require fixed pairwise preference probabilities and conditionally independent responses for each queried edge.

## Quick start
```bash
python -m pip install -r src/cert_voi/requirements.txt
python src/cert_voi/test_joint.py
python src/lookahead/test_finite_horizon.py
python src/cert_voi/benchmark.py
```

The complete 2026-11-16 reviewer-evidence archive (raw JSON results and prior-art notes) is retained alongside the working source in the research conversation. Do not mistake a passing unit test for proof of independent algorithmic novelty.

See `docs/` for the detailed theory and experimental limitations.

Copyright (c) 2026 Ahmad Mohammadi. No open-source license granted.

## 2026-11-16: KL-constrained full-simplex certificate extension

Latest source, finite-action separation proof, real-model pilot, tests and scientific caveats are in
[\`research/full_simplex_20261116/\`](research/full_simplex_20261116/README_FA.md).

This extension proves separation from **single-transfer policy classes**, not from arbitrary unrestricted adaptive methods. The pretrained-LM pilot uses **real model logits but simulated, programmatic preference labels**, not human feedback. Earlier CA-VOI sources and tests remain unchanged.

Reproduce the new stage:

\`\`\`bash
python -m pip install numpy scipy
python -m unittest discover -s research/full_simplex_20261116/tests -v
python research/full_simplex_20261116/strong_adaptive_study.py --ks 6 12 24 48 --n 5000 --seeds 40 --out cyclic_study.json
# On a machine with internet access to Hugging Face weights:
python -m pip install torch 'transformers>=4.47,<5' huggingface_hub
python research/full_simplex_20261116/real_llm_pilot.py --n 5000 --out real_llm.json
\`\`\`

The workflow \`AFA full-simplex verification and real-LM pilot\` runs the checks on GitHub Actions and saves raw experimental JSON as a run artifact.


## Certified algorithm / mathematical proof completion (2026-11-16)

The latest finite-action implementation and proof package is in
[`research/certified_completion_20261116/`](research/certified_completion_20261116/README_FA.md).

Changes:
- Fixed the small-sample confidence radius bug for observations in [-1,1].
- Added a KL-feasible primal policy, a global dual upper bound, and auditable numerical optimality gaps.
- Added a non-anticipating adaptive preference-query and stopping controller.
- Provided five finite-action mathematical statements with proofs, independent grid verification, and regression tests.
- Re-ran 160 synthetic cases and a **real pretrained-model / synthetic-preference-oracle** pilot.

CI evidence: [AFA certified algorithm and proof verification](https://github.com/ahmadmohammadi80/AFA-GP-CIC/actions/runs/43848971749).
It passed 23 tests, 160 synthetic trials, 16 grid audits and 12 real-model questions.

**Scope:** Results are for stationary, conditionally independent finite-action pairwise preferences with a known fixed reference policy. Numerical gaps are float64 checks, not formal directed-rounding certificates. Originality relative to all previous work and validity with human preferences remain separate unverified research questions.


## 2026-11-16 verified numerical/finite-sample closure

See [finite-action closure](research/verified_closure_20261116/README_FA.md)
and [mathematical proof and limitations](research/verified_closure_20261116/PROOF_CLOSURE_FA.md).

- Exact rationalized simplex witnesses and high-precision interval checks for
  KL feasibility, primal lower bound, and dual upper bound; **fail-closed** API.
- Non-implicit, sufficient per-action sample bound and conservative forced
  exploration cost bound for stationary bounded comparisons.
- CI: [30 regression tests passed on main](https://github.com/ahmadmohammadi80/AFA-GP-CIC/actions/runs/43855470942)
  (23 prior and 7 new). This is an audit of the *finite-action stationary*
  model and does not imply global originality or performance with human labels.
- Assumes the interval arithmetic library correctly encloses transcendental
  functions. This is not a separately machine-checked Lean/Coq proof.

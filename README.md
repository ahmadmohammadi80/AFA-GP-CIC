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
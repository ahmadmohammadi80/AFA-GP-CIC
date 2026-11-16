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

# AFA-GP-CIC — reviewer reproduction package

This folder provides a **one-command reproducibility test** of the article's
frozen, stationary synthetic online benchmark. It does not make an empirical
claim of general online superiority, nor a claim about real human / LLM
preferences. The exactness theorem concerns a deterministic two-query tree
on a fixed local uncertainty state; the online gated controller is heuristic.

## Fresh checkout

Clone this PR branch (or the eventual merged commit) and from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -r reviewer/requirements.txt
python reviewer/run.py
```

The script runs the existing **nine** unit tests, generates four JSON datasets
(primary 40, independent holdout 40, cyclic 20, null 20) with three methods
each, and then checks all **360 method-seed scientific records** against the
pinned reference fingerprints. The script exits nonzero if the result differs.

The outputs are saved in `results/reviewer/`, including a machine-readable
`replay_report.json` and `environment.json`. To validate previously generated
output without rerunning:

```bash
python reviewer/run.py --verify-only --results results/reviewer
```

**Pinned environment:** Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
The workflow `.github/workflows/afa-reviewer-reproduce.yml` runs the
same commands on a clean GitHub-hosted Ubuntu runner and publishes outputs.

## Provenance and validation

- Frozen algorithm source commit:
  `498be2bc14b7efa5f88d6f0ff7cae31fba9f3395`.
  This package changes no algorithm implementation or certificate.
- Historical reference CI:
  https://github.com/ahmadmohammadi80/AFA-GP-CIC/actions/runs/44360746204
  (artifact `afa-online-two-step-results`, ID 14456270045; artifact retention
  is finite, so reviewers should generate their own copy).
- The expected per-dataset hashes were computed from historical raw JSON.
  `reviewer/reference_manifest.json` specifies every reference field, seed,
  sample budget, threshold, expected certificate counts and average labels.
- Hashes cover all **13 integer/Boolean/decision-path fields per record**
  (including per-edge query counts); elapsed wall-clock seconds are excluded
  because machines differ. Floating-valued certificate scores are not hashed.
- False certificates, methods, budgets, row order, per-seed identity and
  mean labels are independently checked before a PASS verdict.

## Expected certifications (same 1510-label budget)

| Regime | Greedy | joint_mix | gated AFA |
|---|---:|---:|---:|
| Primary (40 seeds) | 34 | 38 | 34 |
| Independent (40 seeds) | 34 | 37 | 34 |
| Cyclic (20 seeds) | 18 | 20 | 18 |
| Null (20 seeds) | 0 | 0 | 0 |

A successful replay establishes reproducibility of the **tested synthetic
protocol**, not formal proof of novelty, correctness of every general theorem,
real-model effectiveness, or a zero-risk guarantee. All three methods share
the same valid acceptance certificate; `joint_mix` is not an oracle.

## Access for editors and reviewers

This repository is currently **private**. An editor needs authorized read
access or a journal-approved, anonymized archival package. If double-blind
review applies, the editor should receive an anonymized snapshot rather than
a link showing the author's identity. The repository has no open-source
license grant; review access does not itself confer permission to redistribute.

## On a clean GitHub runner

Open **Actions → AFA reviewer reproduction (frozen benchmark)** and run the
workflow from the PR branch. Download `afa-reviewer-scientific-replay`;
the report must contain `all_pass: true` and
`total_scientific_rows: 360` before presenting the package as CI-verified.

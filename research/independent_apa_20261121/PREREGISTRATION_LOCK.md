# Independent human preference validation — frozen prospective APA plan (2026-11-21)

**Stage:** before inspection of APA heldout comparison outcomes. This is a public GitHub timestamped *protocol anchor*, not OSF/AsPredicted registration, nor a completed human experiment.

**Locked full protocol artifact:** `PREREGISTRATION_v1.json` SHA-256 `22e6266dce3332a7379808b44c6dbcbb701bef1bdfb8d4b95753dac60d40c940`.
**Preflight code:** `apa_preflight.py` SHA-256 `e42cb872ddb4be706dea566bab054e9ad01d157eb08266cfd787de313af0c582`.
**Preflight unit tests:** `test_apa_preflight.py` SHA-256 `13b3cf5efdd49cdb81780ce42c536d71339222d7e870081cb4a6df0cc3599229`.

Data: PrefLib APA elections 2000 (`00028-00000003.soi`, Git blob `71a5706358a848e4c647cb925f805e444efb6768`, 11,255 complete ballots) and 2001 (`00028-00000004.soi`, blob `d04322125d151f75a550096d163c6660453c8102`, 10,222 complete ballots). Unlike the internal draft requiring one >=10-alternative collection, these are two different **five-alternative** human election populations. This scope amendment was made before looking at heldout preference outcomes. Do not claim they are independent RLHF comparisons.

Per year: expand complete strict ballot multiplicities in source-row order; shuffle with deterministic unsigned xorshift32 Fisher–Yates, seed = 20261121 + year; take first 5,000, train=first 2,500, untouched holdout=next 2,500. Do not infer missing preferences from partial ballots.

Train-only policy selection: uniform baseline `b`, 20 lexicographically ordered directed 0.09 transfers; use top two maximin training scores, five pure opponents, `KL<=0.05`. **Never select from heldout performance.**

Locked method: original AFA-GP-CIC planner at research branch `research/complete-six-issue-package-20261116`; beta(1/2,1/2) mixture CS at simultaneous `delta/E`. `K=5,E=10,Q=2,H=5,kappa=0.05,gamma=0.015,delta=0.05,eta_gate=0.0005`; `cap=8000` total labels, 10 initial included; methods AFA/Greedy/Joint-Mix/Round-Robin/Width; identical per-edge streams within paired seed.

Experiment 1: full online replay, 40 fixed seeds 20262101..20262140 per method and year, right-censored label-to-certificate and certification count. Fixed cap 8000 is from exploratory SUSHI diagnostics; keep the separate negative 1510-label SUSHI study unaltered.

Experiment 2: two years x 3 origins (Greedy/AFA/Joint-Mix), 1,200 attempted prefix seeds per stratum 20263001..20264200; select *first* state with `0<gamma-maxLCB<=0.0005`, width ratio >=1.5, >=20 label slots remaining. Perform 24 paired continuations at horizons 2,5,10,20. Primary effect: equal-weight six-stratum AFA-minus-Greedy certificate probability at +5 labels; paired prefix-seed-cluster sign-flip p<=0.05 and 95% cluster bootstrap lower bound >0. Require >=300 independent eligible seed clusters overall and >=50 per stratum; else explicitly mark infeasible, no positive inference. Holm corrections for all prespecified secondary comparisons. Population-generalization claim additionally requires respondent-level hierarchical bootstrap with full adaptive replay; otherwise only heldout empirical-kernel conditional claim.

**Protection:** No changes to gamma, Gate, cap, algorithm, years, primary horizon or selection criteria after viewing holdout. Missing snapshots means *no primary estimate*, not zero effect. Report negative findings. Publication contains derived aggregates, not ballots.

**Prior evidence:** The SUSHI 1510-label 2026-11-20 study gave zero certificates and zero AFA activations; the 8000-label SUSHI Greedy reachability check was post hoc, not independent evidence. 

The canonical full JSON and tested implementation accompanying this anchor are provided in the project conversation as a downloadable package. The three hashes cryptographically bind their bytes; any later alteration must be labeled a new protocol version.

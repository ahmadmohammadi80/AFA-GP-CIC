# AFA-GP-CIC: six-issue formal gate, starting with issue 1

Status: OPEN. We do not claim independent novelty, sample improvement, or completion of issues 1-6.

Root cause before modification: the old one-step recourse objective maximizes the expected numerical value of the next certificate rather than minimizing label cost to reach positive confidence. With 50-label batches it cannot stop between labels; but after equal-clock correction the joint algorithm still certified 13/40 versus 36/40 for beta-mixture batch greedy. Hence cadence alone does not explain failure.

Three controlled per-label scheduling alternatives were implemented with the same KL policies, opponents, stationary pairwise Bernoulli streams, anytime beta-mixture confidence intervals and label budget: joint_active retains plausible alternatives; joint_mix combines currently best-policy greedy with multi-policy exploration; joint_gap weights candidate-specific bottlenecks by distance to the acceptance threshold. Each may waste labels on implausible candidates. The baseline is the existing strong greedy queried per label. Primary 40 seeds are offset 20261116; a separate 40 seeds at 20270101 are held out. Improvement cannot be accepted only on tuned seeds.

Other gates remain OPEN: (2) two genuine literature baselines and multiplicity-controlled paired tests; (3) new full stopping-objective theorem and matching constructive nontransitive witness; (4) bound for actual full algorithm and clear minimax comparison, without generalizing the two-known-world 1.63 result; (5) proved drift or continuous-model certificate with calibration; (6) scaling, profiling and measured optimization. No manuscript-ready claim is licensed merely by passing unit tests.

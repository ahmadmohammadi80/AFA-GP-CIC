# Time-uniform policy-reusable certificate — scoped extension

## New guarantee in this codebase (not a globally novel concentration inequality)

Let E be the total number of unordered pairwise edges. For every edge e, an infinite sequence of independent bounded signs Y_(e,s) in {-1,+1} has fixed mean mu_e. The algorithm can adaptively choose the next edge, candidate policy q, opponent set H, and stopping time, using all previously observed signs. Denote n_e the number of observations on edge e, and the prefix mean m_e(n_e).

Define r(n)=sqrt(2 log(E*pi^2*n^2/(3*delta))/n). Hoeffding gives P(|m_e(n)-mu_e|>r(n)) <= 2 exp(-n*r(n)^2/2)=6 delta/(E*pi^2*n^2). Sum over all e and all positive integers n gives <=delta. Therefore simultaneously for **all** q and H, even selected post hoc from observed data, the certificate

L(q,H)=min_(h in H) sum_e a_(q,h,e)*m_e(n_e)-sum_e |a_(q,h,e)|*r(n_e)

is <= the true min_(h in H) (q-b)^T W h with probability >=1-delta.

This removes the independent confirmation split needed by the previous fixed-time method and permits arbitrary adaptive stopping and candidate choice **under stationary independent per-edge feedback**. It does not imply a sample-complexity advantage over best known anytime confidence sequence methods; the union-bound radius is conservative. If an edge is never sampled, the current implementation requires an initial one sample per edge; this can be replaced by the vacuous [-1,1] interval without affecting the proof.

## Critical scope and novelty

This is a correct standard union-bound/anytime argument, **not** an independent global novelty result. The *candidate* mechanism is coupling reusable simultaneous preference confidence sets to a KL-feasible adaptive policy selector, but that coupling overlaps known confidence-sequence and active learning principles. No claim of original rate, minimax separation, or superiority to Robust Nash, Nash Learning, POSTPI or active preference learning is justified by this result alone.

## Fair comparator

A fair baseline may use the same observations, candidate policies, KL constraint, adaptive edge selection and anytime validity; only the query allocation / decision rule may differ. Comparing against an independent-confirmation-only baseline alone would confound the effect of data reuse. The synthetic smoke test checks code invariants and false-certification, not superiority. A future gate requires an instance-specific lower bound against a strong adaptive comparator and matched total labels.

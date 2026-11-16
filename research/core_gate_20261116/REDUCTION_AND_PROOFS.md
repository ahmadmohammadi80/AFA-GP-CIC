# Core decision proofs (research note, not manuscript)

Frozen base: fa5b5b20ad8d7d9cf264d1c7e000b61c1afa6469.

## Exact problem reduction

Fix finite Q,H, baseline b, threshold tau, and a KL-feasible Q chosen before observations. For each unordered edge e=(i,j), query X_e~Bernoulli(p_e), and define W_ij=2p_e-1, W_ji=-W_ij. Set

 a(q,h,e)=(q_i-b_i)h_j-(q_j-b_j)h_i.

Expanding the matrix product by unordered edges gives

 g_p(q)=min_h (q-b)^T W h=min_h sum_e a(q,h,e)(2p_e-1).

Thus each edge is one Bernoulli arm; each candidate q is one possible answer; a correct answer has g_p(q)>tau. The set where q is wrong is

 Alt(q)=union_h {p: 2 sum_e a(q,h,e)p_e <= tau+sum_e a(q,h,e)}.

This is a union of halfspaces intersected with the mean-parameter domain. Skew-symmetry and cycles do not obstruct the mapping. A fixed KL constraint only restricts the known answer set Q; it does not introduce a new feedback model.

Degenne and Koolen (2019), Sections 2–5, already define finite-answer identification over exponential-family arms, answer-dependent alternatives, and adaptive joint answer/allocation selection. Source:
https://proceedings.neurips.cc/paper/2019/file/60cb558c40e4f18479664069d9642d5a-Paper.pdf

Our reduction is algebraic, not a proof that the present implementation equals Sticky Track-and-Stop. Their fixed-confidence completion guarantee and asymptotic expected stopping-time criterion must not be transferred to a finite-budget AFA procedure that can abstain. For a complete identification formulation add a “none” answer on strictly negative instances and exclude relevant zero-margin boundaries. The positive-answer alternatives above then apply within that model domain. The associated published information-design expression specializes to

 max_(valid q) max_(w in simplex) min_h inf_(p' in Alt(q,h)) sum_e w_e kl(p_e,p'_e).

No claim is made that the existing 0.7/0.3 heuristic solves this expression, that the formula is new, or that an asymptotic theorem by itself ranks algorithms at B=24. We did not implement full Sticky-TaS in this gate.

## Fair finite-budget counterexample

Take b=(1/3,1/3,1/3), a=0.12, q1=b+a(e1-e0), q2=b+a(e2-e0), and H={e0}. Their common KL is below .05. Let p1,p2 be the win probabilities on edges (0,1),(0,2). Then g_p(q_i)=a(1-2p_i). Edge (1,2) has zero coefficient in both answers; give this exact support reduction to EVERY method. Both receive independent Beta(1/2,1/2) planning priors, one initial sample per active edge, the same confidence construction, delta=.05, tau=.015, and B=24 total labels. Every twentieth post-pilot query is a shared predetermined round-robin coverage query.

In state s=(n1,x1,n2,x2), the certificate is the original split-delta beta-mixture/Hoeffding intersection applied to each active edge, and terminal C(s) is true if either policy lower bound exceeds tau. This is a proper instance of the finite-policy AFA objective. It does not reproduce the five-action historical experiment.

## Standard finite-horizon redesign reference

Let r=B-n1-n2 and E(s) be the allowed query set (a singleton on forced-coverage rounds). With the common planning prior, predicted win probability on edge e is (x_e+1/2)/(n_e+1). Define

 V(s)=1 if C(s),
 V(s)=0 if r=0 and not C(s),
 V(s)=max_(e in E(s)) [(1-pred_e)V(s+failure_e)+pred_e V(s+success_e)] otherwise.

Induction on r proves optimality among all randomized or deterministic history-dependent strategies sharing this model, terminal event, prior, forced schedule and budget: it is true at r=0; conditional expectation at each action is bounded by the recurrence using the induction hypothesis; a maximizing action attains it. Randomizing actions cannot exceed their maximum. This is the standard Bellman principle, not an independent innovation.

AFA's current score rule does not optimize this quantity. The supplied exact rational recursion, with independently verified acceptance flags, yields

 V_joint=1514620435893/4398046511104,
 V_greedy=3283851329151/8796093022208,
 V_Bellman=7090878750435/17592186044416.

Hence V_joint<V_greedy<V_Bellman strictly. These are prior-averaged probabilities of the COMMON certificate event. They are not pointwise guarantees and not p-values. In particular Bellman does not beat greedy on every fixed p in the complete reported 49-point grid. The generic engine's support-aware score and the counterexample implementation agree on tested nonterminal states; the candidate protocol is not secretly given true p.

## Independent rational check of the terminal event

For one edge, the safety probability threshold is p0=(1-tau/a)/2=7/16. At n<=24 and every 0<=x<=n, compute exactly

 M_n(p0)=(1/2)_x (1/2)_(n-x) / [(1)_n p0^x (1-p0)^(n-x)].

Here (z)_n is a rising factorial. The beta-mixture upper endpoint lies below p0 exactly when x/n<p0 and M_n(p0)>2E/delta=80. Each expression is rational. All 324 acceptance flags are compared with the implemented intersection; for this horizon the Hoeffding branch adds no extra acceptance. Thus certificate threshold classification in this example does not depend on trusting numerical root location. Heuristic query choices still use the frozen floating-point rule; exact probabilities are conditional on those deterministic choices. The Bellman probability recursion uses rational probabilities and maximization independently.

## Decision and research boundary

The algebraic reduction defeats the claim that merely combining policy choice with querying creates a new problem class. The executable counterexample defeats universal dominance of the current score. Neither proves that no new algorithm for this problem can exist. A redesigned practical rule would need an explicit finite-budget approximation guarantee, computational cost, and comparison with this exact reference on small problems and a faithful literature method on larger ones. That contribution has NOT been obtained here. The six-issue task remains scientifically open; no manuscript-ready or globally original algorithm claim is licensed.

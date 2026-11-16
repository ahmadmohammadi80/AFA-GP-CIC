# CSWM: a valid local expert-regret theorem, and its limits

This is an application of EXP3-IX (Neu, NeurIPS 2015), not a new regret theorem.
Primary source: https://proceedings.neurips.cc/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf . The derivation below specializes its concentration argument and records the comparator explicitly.

## 1. Certificate sensitivity and feedback

Fix the finite coefficient tensor A(q,h,e), signed means W_e in [-1,1], and confidence intervals [l_e,u_e]. Set

C = 2 max_(q,h) sum_e |A(q,h,e)|,
U(s) = max_(q,h) sum_e |A(q,h,e)| (u_e-l_e) / C.

If C=0, define U=0. Otherwise 0<=U<=1. This is a normalized sensitivity envelope, NOT a probability that a certificate is wrong. For L_q=min_h sum_e min(A l,A u) and V_q=min_h sum_e max(A l,A u), V_q-L_q <= C U. Indeed, min_h b_h-min_h a_h <= max_h(b_h-a_h). Also max_q V_q-max_q L_q <= max_q(V_q-L_q). Thus U bounds the width of the best-policy value interval, with scale C. It need not measure proximity to the acceptance threshold.

At round t all K experts propose an allowed edge before expert I_t is sampled. On the realized shared state s_t define a potential label Y_(e,t) for each edge, generated independently of the current random expert choice conditional on history. This is an analytical coupling; the learner observes only Y_(e_I,t). The potential reward is

r_(i,t) = U(s_t) - U(s_t updated with (e_(i,t),Y_(e_i,t))).

Consequently r_(i,t) is in [-1,1], including when intervals widen or known drift enlarges stale-edge bounds. Define loss l_(i,t)=(1-r_(i,t))/2 in [0,1]. Proposals may be randomized (Thompson); condition on their draws before sampling I_t. The environment may depend on the past, but not on the still-hidden current expert choice. Identical proposed edges share the same potential label.

A label from one edge does not reveal the label of another. No simulated counterfactual is treated as observed. The implementation's `estimate_counterfactual` returns None on a different edge. Bayesian predictive counterfactuals would define a model-dependent surrogate, not the same realized-feedback theorem.

## 2. Update and fixed-horizon theorem

Let p_(i,t) be normalized exponential weights, eta>0, gamma=eta/2, initially uniform. Use

ltilde_(i,t) = 1{I_t=i} l_(I_t,t)/(p_(i,t)+gamma),
log w_(i,t+1) = log w_(i,t) - eta ltilde_(i,t).

This is EXP3-IX, with an expert index as the bandit action. It differs deliberately from uncorrected updates of all experts using fabricated labels. The user's expression U_before-U_after is a reward; applying a negative exponential directly to it would favor the wrong direction.

For K>=2, with probability at least 1-delta_R, simultaneously over expert j and all prefixes t<=H,

R_reward(t) = max_j sum_(s<=t) r_(j,s) - sum_(s<=t) r_(I_s,s)
 <= 2[(log K + log(2K/delta_R))/eta + eta K t + log(2/delta_R)].

Choosing eta=sqrt((log K+log(2K/delta_R))/(K H)) gives, at H,

R_reward(H) <= 4 sqrt(K H [log K+log(2K/delta_R)]) + 2 log(2/delta_R).

For fixed K=3 or 4 this is sublinear in H. Dependence on K and delta must not be suppressed when they vary. For H=22 this worst-case upper bound is vacuous (larger than the trivial maximum 44); it is not a short-budget performance guarantee.

### Proof of the needed concentration inequality

Write beta=2 gamma. For x>=0, log(1+x)>=2x/(2+x). With x=beta l_i/p_i and l_i<=1, this gives

beta l_i/(p_i+gamma) <= log(1+beta l_i/p_i).

For predictable alpha_i in [0,beta], concavity of z^(alpha_i/beta) implies

exp(alpha_i l_i/(p_i+gamma)) <= 1+alpha_i l_i/p_i.

Only the selected coordinate of ltilde is nonzero. Therefore, conditionally on the potential loss vector and past,

E exp(sum_i alpha_i(ltilde_i-l_i))
 <= exp(-sum_i alpha_i l_i) [1+sum_i alpha_i l_i] <= 1.

Multiplying these conditional bounds constructs a nonnegative supermartingale starting at one. Markov's inequality gives the fixed-time bound; the maximal supermartingale inequality gives all-prefix validity. Use alpha_i=2gamma 1{i=j}, and union over K experts, to obtain with probability 1-delta_R/2

sum_s(ltilde_(j,s)-l_(j,s)) <= log(2K/delta_R)/(2gamma), all j, all prefixes.

Use alpha_i=2gamma for all i to obtain, on another event of probability 1-delta_R/2,

sum_(s,i) ltilde_(i,s) <= Kt + log(2/delta_R)/(2gamma), all prefixes.

### Exponential-weights potential

The standard inequalities exp(-x)<=1-x+x^2/2 for x>=0 and log(1+x)<=x yield, for every j,

sum_s p_s dot ltilde_s - sum_s ltilde_(j,s)
 <= log K/eta + (eta/2) sum_(s,i) p_(i,s) ltilde_(i,s)^2.

Because l_i<=1 and only I_s is nonzero,

sum_i p_i ltilde_i^2 <= sum_i ltilde_i,
l_(I_s,s) = p_s dot ltilde_s + gamma sum_i ltilde_(i,s).

Substitute both concentration events into the potential bound. Since eta/2+gamma=eta and 2gamma=eta,

R_loss <= log K/eta + eta Kt + log(2/delta_R) + log(2K/delta_R)/eta.

Reward regret is twice loss regret. This proves the theorem. Ordinary Azuma applied naively to inverse-probability estimates would not prove it: their range is not uniformly bounded as selection probabilities approach zero.

## 3. Anytime implementation

A supplied fixed eta only proves the eta-dependent bound above, which contains a linear eta Kt term. To obtain an unknown-horizon sublinear guarantee, instantiate `eta=None`. Epoch m has H_m=2^m rounds, resets weights to uniform, uses delta_m=6 delta_R/[pi^2(m+1)^2] and the tuned eta_m. The all-prefix proof applies inside every epoch; sum delta_m=delta_R. Sum the epoch bounds, including the incomplete last epoch. For all T simultaneously,

R_reward(T) <= sum_(m<=floor(log2 T)) {4 sqrt(K H_m [log K+log(2K/delta_m)]) + 2 log(2/delta_m)}.

The leading term is O(sqrt(K T [log K+log(1/delta_R)+log log(T+2)])); additive confidence terms are O(log(T+2)[log(1/delta_R)+log log(T+2)]). This is sublinear, but is not the user's unqualified uniform O(sqrt(T log K)) claim. Primary finite-budget experiments use fixed H=22 and fixed eta, with no resets.

## 4. What this does NOT prove

* The comparator uses each expert's one-step recommendation at the learner's actual states. It does not run each expert on its own counterfactual state history. That stronger policy-regret guarantee does not follow.
* There is no finite-time guarantee of matching or exceeding the best expert. A positive regret allowance remains, and high probability is not certainty.
* Cumulative uncertainty reduction along an actual run telescopes to U(s_start)-U(s_end). The terminal event of crossing tau is a different objective. Optimizing this envelope can reduce irrelevant widths without issuing a certificate.
* Static-expert regret is not regret against a changing best expert. Arbitrary drift does not imply weight concentration on the currently best strategy. Resets in the anytime schedule do not establish a switching-regret theorem.
* Certificate validity needs its own delta_C and a valid CS model. Simultaneous certificate validity and this regret theorem have probability at least 1-delta_C-delta_R, unless a total confidence budget is explicitly split. Here both are .05, so the union statement is .90, not .95.
* Bayesian full-budget Bellman is optimal for its specified Beta prior objective and allowed actions. It does not know the true world p; calling it a true-world oracle is incorrect. No adaptive randomized mixture with the same information can improve that prior objective beyond its optimum. It need not dominate in every fixed world.

## 5. Complexity and novelty

The generic sensitivity envelope costs O(Q H E). Computing one hypothetical state and envelope is at least this costly in the unoptimized implementation. A depth-d binary-outcome planner branches over allowed edges, costing approximately O((2E)^d Q H E), or a state-dependent DP cost with caching. Thus O(K Q H E) is NOT a valid general complexity for experts containing Bellman planning. Full remaining-horizon planning is much more expensive. Storing A costs O(Q H E); K weights cost O(K). The small gate uses cached sufficient-count states. Runtime in results is for this small benchmark and is not large-scale evidence.

Prior work already combines active learners with bandit allocation and entropy feedback: Baram, El-Yaniv & Luz, JMLR 2004, https://www.jmlr.org/papers/v5/baram04a.html . Active Learning by Learning (Hsu & Lin, AAAI 2015) also mixes query strategies using bandits: https://ojs.aaai.org/index.php/AAAI/article/view/9597 . EXP3-IX supplies the meta-regret analysis. A certificate-specific feedback application is implemented here; independent methodological novelty is UNVERIFIED. This targeted search is not an exhaustive systematic review and cannot support an absence-from-literature assertion.

# AFA-CAC: certificate-aware edge allocation — 16 November 2026

Status: executable candidate method and provable restricted allocation; **global originality not established**. No manuscript.

## Core decision
For baseline b, fixed KL-feasible q and opponent distributions h^r, skew preference matrix W, write d=q-b. For edge e=(i,j), i<j:

  a_(r,e) = d_i h^r_j - d_j h^r_i.

Then g_r=d^T W h^r=sum_e a_(r,e) W_ij. Only relevant edges need comparison. This representation keeps cyclic information: replacing W by W+C where Cb=0 changes g_r for general h^r, even though Wb is unchanged.

For E active edges and independent fixed-size confirmation batches N_e, c=sqrt(2 log(2E/delta)). Standard Hoeffding on Y in [-1,1] implies:

  L = min_r (sum_e a_(r,e) mean_e - c sum_e |a_(r,e)|/sqrt(N_e))

is a valid lower bound to min_r g_r with simultaneous probability >=1-delta.

Pilot samples may determine q, opponents and N, **provided confirmation samples are independent of the pilot** and the final reported q,H remain fixed during the confirmation stage. A model that changes q after seeing confirmation samples needs additional simultaneous-policy or anytime guarantees.

## Theorem 1 — exact one-opponent allocation under rectangular confidence widths

For fixed nonnegative weights w_e and target B>0, minimize sum N_e subject to sum w_e/sqrt(N_e)<=B over real N_e>0. The global optimum is:

  S=sum_e w_e^(2/3);
  N_e=(S/B)^2 w_e^(2/3);
  N_total=S^3/B^2.

Proof: KKT gives 1=lambda*w_e/(2*N_e^(3/2)); enforcing the tight constraint determines lambda. Slater and convexity guarantee global optimality; integer ceiling is feasible and adds < E labels. This is NOT a lower bound for arbitrary adaptive sampling and the 2/3 power allocation is not novel by itself.

## Theorem 2 — convex multi-opponent allocation and auditable dual

For weights A_(r,e), simultaneously require sum_e |A_(r,e)|/sqrt(N_e)<=B_r. This is a convex program. For ANY nonnegative multipliers lambda_r, weak duality gives:

  N_opt >= 3*2^(-2/3)*sum_e (sum_r lambda_r*|A_(r,e)|)^(2/3) - sum_r lambda_r*B_r.

Proof: minimize n+z/sqrt(n) edge by edge at n=(z/2)^(2/3). This bound is valid without numerical optimization, but the current implementation records only a guaranteed conservative lower bound obtained from one active opponent and does NOT claim full multi-constraint numerical optimality.

## Test methodology

- The pilot collects labeled comparisons but NEVER appears in the final confidence interval.
- Pilot labels choose confirmation counts; fresh comparison labels independently produce the certificate.
- Uniform baseline has exactly the same pilot, q, opponent set, relevant edge set and confirmation comparison budget.
- Both baselines use their own fresh confirmation labels.
- Report the number of skipped pilot trials; no success-rate conclusion without it.
- The current method designs *counts for an already fixed policy*, and has not solved the joint q-and-query optimization problem.

## Comparison to published work

Prior work includes Robust Nash Alignment under Preference Uncertainty (Ahmed et al., arXiv:2610.00715), Efficient Exploration for Iterative Nash Preference Optimization (Nan et al., arXiv:2606.01382), and Pure Exploration in Bandits with Linear Constraints (Carlsson et al., AISTATS 2024). These already treat robust Nash games, active exploration, or optimal constrained allocations respectively. Thus our standalone ingredients and restricted theorems do not establish global novelty. The next necessary contribution is a jointly adaptive q/H/query scheme with an instance-dependent **information-theoretic** advantage over a strong matched-information prior-art-aware comparator.

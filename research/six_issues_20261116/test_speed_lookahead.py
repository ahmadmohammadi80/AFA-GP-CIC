import unittest
import numpy as np
from speed_lookahead import recourse_fast,frontier_fast,dense_recourse,dense_frontier
class SpeedTests(unittest.TestCase):
 def test_exact_lookahead(self):
  for k in (3,5):
   rng=np.random.default_rng(100+k);b=rng.dirichlet(np.ones(k));Q=rng.dirichlet(np.ones(k),size=3)
   n=np.full(k*(k-1)//2,30,int);w=rng.binomial(n,.5);H=np.eye(k)
   a=dense_recourse(b,Q,H,w,n,.05,5);v=recourse_fast(b,Q,H,w,n,.05,5)
   for field in ("recourse","precommit","current_policy_certificates"):
    np.testing.assert_allclose(a[field],v[field],atol=1e-12,rtol=0)
   self.assertEqual(a["best_joint_edge"],v["best_joint_edge"])
 def test_frontier(self):
  for k in (3,6,12):
   rng=np.random.default_rng(k);b=rng.dirichlet(np.ones(k))
   for kl in (0,.05,.5,10):
    np.testing.assert_allclose(frontier_fast(b,kl),dense_frontier(b,kl),atol=1e-11,rtol=0)
if __name__=="__main__":unittest.main()

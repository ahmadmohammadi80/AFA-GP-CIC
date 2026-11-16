import unittest
import numpy as np
from pure_opponent_fast import fast_pure,dense_reference
class ScalingTests(unittest.TestCase):
 def test_fast_matches_dense_all_dimensions(self):
  for k in (2,3,5,10,20):
   rng=np.random.default_rng(k);b=rng.dirichlet(np.ones(k))
   Q=rng.dirichlet(np.ones(k),size=4);E=k*(k-1)//2
   c=rng.uniform(-.7,.7,E);r=rng.uniform(.01,.2,E)
   lo=np.maximum(-1,c-r);hi=np.minimum(1,c+r)
   np.testing.assert_allclose(fast_pure(b,Q,lo,hi),dense_reference(b,Q,lo,hi),atol=1e-12,rtol=0)
 def test_invalid_interval(self):
  with self.assertRaises(ValueError):fast_pure([.5,.5],[[.7,.3]],[.3],[.2])
if __name__=="__main__":unittest.main()

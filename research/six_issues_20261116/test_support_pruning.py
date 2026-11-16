import unittest
import numpy as np
from support_pruning import coefficients,support,certificate,demonstration
class TestSupport(unittest.TestCase):
 def test_exact_certificate_invariance(self):
  rng=np.random.default_rng(51)
  for k in (6,10,20):
   b=np.zeros(k);b[:2]=.5
   Q=np.array([b.copy()]);Q[0,0]+=.1;Q[0,1]-=.1
   H=np.eye(k)[2:4];_,A=coefficients(b,Q,H);S=support(A)
   lo=rng.uniform(-1,0,A.shape[-1]);hi=rng.uniform(0,1,A.shape[-1])
   original=certificate(A,lo,hi)
   compact=certificate(A[:,:,S],lo[S],hi[S])
   np.testing.assert_allclose(original,compact,rtol=0,atol=1e-14)
 def test_sparse_witness(self):
  d=demonstration(n=2000)
  self.assertEqual(d["active_edges"],4)
  self.assertGreater(d["uniform"]["queried_inactive"],0)
  self.assertEqual(d["support_only"]["queried_inactive"],0)
  self.assertGreater(d["support_only"]["minimum_active_count"],d["uniform"]["minimum_active_count"])
if __name__=="__main__":unittest.main()

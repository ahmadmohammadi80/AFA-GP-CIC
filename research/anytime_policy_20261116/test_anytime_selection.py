import unittest
import numpy as np
from anytime_selection import radius,edge_coefficients,reusable_lower,sample_run
class AnytimeTests(unittest.TestCase):
 def test_radii_decrease_eventually(self):
  r=radius(np.array([10,100,1000,10000]),.05,10)
  self.assertTrue(np.all(np.diff(r)<0))
 def test_coefficients(self):
  rng=np.random.default_rng(2)
  b=np.ones(5)/5;q=rng.dirichlet(np.ones(5));H=np.eye(5)
  X=rng.uniform(-.4,.4,(5,5));W=X-X.T
  edges,A=edge_coefficients(b,q,H)
  np.testing.assert_allclose(A@np.array([W[i,j] for i,j in edges]),[(q-b)@W@h for h in H])
 def test_time_uniform_event(self):
  rng=np.random.default_rng(1);mu=.21;E=5
  for _ in range(30):
   seq=2*rng.binomial(1,(1+mu)/2,size=2000)-1
   n=np.arange(1,2001)
   self.assertTrue(np.all(abs(np.cumsum(seq)/n-mu)<=radius(n,.05,E)))
 def test_reuse_after_selection(self):
  rng=np.random.default_rng(12);k=4;b=np.ones(k)/k
  Q=[rng.dirichlet(np.ones(k)) for _ in range(15)]
  H=np.eye(k)
  X=rng.uniform(-.3,.3,(k,k));W=X-X.T
  edges=[(i,j) for i in range(k) for j in range(i+1,k)]
  n=np.full(len(edges),500)
  means=np.array([2*rng.binomial(500,(1+W[i,j])/2)/500-1 for i,j in edges])
  for q in Q:
   self.assertLessEqual(reusable_lower(b,q,H,means,n),min((q-b)@W@h for h in H)+1e-10)
 def test_adaptive_loop(self):
  for s in range(10):
   r=sample_run(seed=s,max_rounds=250)
   self.assertLess(r['KL'],.05)
   self.assertGreaterEqual(r['samples'],10)
   if r['certified']:self.assertGreater(r['true_gain'],r['threshold'])
 def test_reject_invalid(self):
  with self.assertRaises(ValueError):radius([0],.05,1)
  with self.assertRaises(ValueError):radius([1],1.,1)
if __name__=="__main__":unittest.main()

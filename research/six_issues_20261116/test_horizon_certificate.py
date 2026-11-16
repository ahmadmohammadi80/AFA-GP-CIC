import unittest
import numpy as np
from horizon_certificate import oracle,greedy,exhaustive,audit,width
class HorizonTests(unittest.TestCase):
 def test_exact_vs_exhaustive(self):
  r=np.random.default_rng(20261116)
  for _ in range(80):
   A=r.uniform(0,2,(3,3));n=r.integers(1,6,3);c=r.uniform(.2,1.5,3)
   o=oracle(A,n,c,4)
   self.assertAlmostEqual(o["minimax_width"],exhaustive(A,n,c,4),places=10)
   self.assertEqual(sum(o["allocation"]),4)
 def test_strict_greedy_counterexample(self):
  A=np.array([[2.1234350277826772,1.0100613894450516,.45147158669871923,1.0384154126809702],
              [2.8005144095893377,2.390035208470793,.1290554594849187,.6769063298407779],
              [.9500297109650852,2.8359055637271493,1.2699674602287003,.6272666800474009]])
  o=oracle(A,np.ones(4,int),np.ones(4),6)
  g=greedy(A,np.ones(4,int),np.ones(4),6)
  self.assertLess(o["minimax_width"],g["minimax_width"]-0.08)
  self.assertEqual(o["allocation"],[2,2,1,1])
 def test_validation(self):
  with self.assertRaises(ValueError):oracle(np.ones((2,3)),np.ones(3,int),np.ones(3),-1)
if __name__=="__main__":unittest.main()

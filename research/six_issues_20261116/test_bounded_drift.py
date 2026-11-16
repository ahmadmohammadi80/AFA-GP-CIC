import unittest
import numpy as np
from bounded_drift import bound,experiment
class DriftTests(unittest.TestCase):
 def test_vacuous_no_samples(self):
  self.assertEqual(bound(0,0,0,3,.1,.05,10),(-1.,1.))
 def test_more_drift_never_narrows(self):
  a=bound(23,40,800,50,0,.05,10)
  b=bound(23,40,800,50,.01,.05,10)
  self.assertLessEqual(b[0],a[0]);self.assertGreaterEqual(b[1],a[1])
 def test_adaptive_simulation(self):
  for s in range(3):
   out=experiment(s,cycles=100)
   self.assertTrue(out["covered_all_pairs_all_times"])
 def test_no_bad_future(self):
  with self.assertRaises(ValueError):bound(2,5,99,10,.01,.05,10)
if __name__=="__main__":unittest.main()

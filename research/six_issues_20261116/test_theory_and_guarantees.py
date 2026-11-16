import unittest
import numpy as np
from theory_and_guarantees import cyclic_witness,guarded_joint,clipped_intersection,theoretical_sample_cap
from published_baselines import preference_problem
class TheoryTests(unittest.TestCase):
 def test_witness_is_truly_cyclic_and_separates_worlds(self):
  v=cyclic_witness();self.assertGreater(v["minimum_true_gain_for_correct_policy"],.04)
  self.assertLess(v["n_diagnostic_comparisons"],40)
 def test_guarded_validity_and_rate(self):
  for seed in range(3):
   v=guarded_joint(seed,budget=410,explore_period=10)
   self.assertFalse(v["false_certificate"])
   self.assertTrue(v["within_bound"])
 def test_explicit_sample_cap(self):
  b,Q,H,W=preference_problem()
  g=[min(float((q-b)@W@h) for h in H) for q in Q]
  v=theoretical_sample_cap(b,Q,H,.015,g)
  self.assertTrue(v["positive_margin"]);self.assertGreater(v["sample_upper"],0)
 def test_intersection_nonempty(self):
  lo,hi=clipped_intersection(30,50,.05,10)
  self.assertLessEqual(lo,hi)
if __name__=="__main__":unittest.main()

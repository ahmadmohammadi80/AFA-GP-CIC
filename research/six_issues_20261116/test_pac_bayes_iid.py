import unittest
import numpy as np
from pac_bayes_iid import radii,run
class PBTests(unittest.TestCase):
 def test_radius(self):
  self.assertGreater(radii(1,.05,10),radii(3000,.05,10))
 def test_deterministic(self):
  self.assertEqual(run(1,budget=210),run(1,budget=210))
 def test_valid_noncyclic_and_cyclic_smoke(self):
  for cyclic in (False,True):
   for seed in range(4):
    r=run(seed,budget=300,cyclic=cyclic)
    self.assertFalse(r["false_certificate"])
    self.assertLessEqual(r["labels"],300)
if __name__=="__main__":unittest.main()

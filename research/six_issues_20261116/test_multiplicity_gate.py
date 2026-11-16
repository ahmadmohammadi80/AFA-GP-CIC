import unittest
from multiplicity_gate import holm
class MultiplicityTests(unittest.TestCase):
 def test_holm_monotonic_order(self):
  ps={"a":.0001,"b":.03,"c":.05,"d":.8}
  adj=holm(ps)
  self.assertAlmostEqual(adj["a"],.0004)
  self.assertAlmostEqual(adj["b"],.09)
  self.assertAlmostEqual(adj["c"],.10)
  self.assertAlmostEqual(adj["d"],.8)
 def test_reject_invalid(self):
  with self.assertRaises(ValueError):holm({"bad":-1})
if __name__=="__main__":unittest.main()

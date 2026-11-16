import unittest
from extended_power import audit
class ExtendedTests(unittest.TestCase):
 def test_small(self):
  r=audit(seeds=4,offset=29990101)
  self.assertEqual(len(r["rows"]["greedy"]),4)
  self.assertEqual(len(r["rows"]["joint_mix"]),4)
  self.assertTrue(0<=r["matched"]["p_power_holm_two"]<=1)
if __name__=="__main__":unittest.main()

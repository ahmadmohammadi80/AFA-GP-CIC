import unittest
from guarded_holdout import audit
class GuardedHoldoutTests(unittest.TestCase):
 def test_small(self):
  r=audit(seeds=3,offset=20300101)
  self.assertEqual(len(r["rows"]["guarded"]),3)
  self.assertEqual(r["summary"]["guarded"]["regret_bound_violations_observed"],0)
if __name__=="__main__":unittest.main()

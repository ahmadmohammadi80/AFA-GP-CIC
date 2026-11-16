import unittest
from change_point_reset import trial,worlds
class ChangePointTests(unittest.TestCase):
 def test_genuine_cycle_and_positive_policies(self):
  b,Q,H,Ws=worlds()
  for z,W in enumerate(Ws):
   self.assertGreater(min(float((Q[z]-b)@W@h) for h in H),.04)
 def test_reset_exactly_and_reproduce(self):
  a=trial(101,epochs=2,epoch_budget=320)
  self.assertEqual(a,trial(101,epochs=2,epoch_budget=320))
  self.assertTrue(a["path_covered_all_epochs"])
  self.assertTrue(all(not x["false_certificate"] for x in a["epoch_rows"]))
if __name__=="__main__":unittest.main()

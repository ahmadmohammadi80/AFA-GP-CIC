import unittest
from issue1_ablation import trial,audit
class Tests(unittest.TestCase):
 def test_reproducible(self):
  for m in ("greedy","joint_active","joint_mix","joint_gap"):
   a=trial(101,m,300);self.assertEqual(a,trial(101,m,300))
   self.assertFalse(a["false_certificate"])
 def test_null(self):
  for m in ("greedy","joint_active","joint_mix","joint_gap"):
   for seed in range(3):self.assertFalse(trial(seed,m,200,null=True)["certified"])
 def test_audit(self):
  r=audit(2,20261116,100);self.assertEqual(len(r["rows"]),4)
if __name__=="__main__":unittest.main()

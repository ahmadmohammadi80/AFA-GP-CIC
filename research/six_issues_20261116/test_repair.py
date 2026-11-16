import unittest
import numpy as np
from guarded_engine import GuardedEngine
from published_baselines import choose_pair,preference_problem
from theory_and_guarantees import guarded_joint
from issue1_ablation import trial

class RepairTests(unittest.TestCase):
 def test_rucb_does_not_promote_random_champion(self):
  _,memory=choose_pair('rucb',np.ones(3,int),np.full(3,2),3,10,np.random.default_rng(3))
  self.assertIsNone(memory)
 def test_rucb_memory_has_exact_half_probability(self):
  rng=np.random.default_rng(9);seen=0
  # All candidates plausible, known singleton memory from an earlier round.
  # An instrumented RNG records the pool used when the memory coin loses.
  class RNG:
   def random(self):return .9
   def choice(self,pool):
    if len(pool)==2 and set(pool)=={1,2}:return pool[0]
    return pool[0]
  e,m=choose_pair('rucb',np.ones(3,int),np.full(3,2),3,10,RNG(),0)
  self.assertEqual(m,0)
  self.assertIn(e,(0,2))
 def test_budget_is_not_silently_exceeded(self):
  for f in (guarded_joint,trial):
   with self.assertRaises(ValueError):f(1,budget=0)
 def test_generic_dense_incremental_identical_trajectory(self):
  rng=np.random.default_rng(42);k=7;b=np.ones(k)/k
  Q=.8*b+.2*rng.dirichlet(np.ones(k),size=6)
  H=rng.dirichlet(np.ones(k),size=4)
  a=GuardedEngine(b,Q,H,threshold=1);d=GuardedEngine(b,Q,H,threshold=1,incremental=False)
  for t in range(400):
   ea,ed=a.next_edge(),d.next_edge();self.assertEqual(ea,ed)
   y=int(rng.integers(2));a.observe(ea,y);d.observe(ed,y)
   np.testing.assert_allclose(a.bounds()[0],d.bounds()[0],atol=2e-14,rtol=0)
 def test_tiny_budget_and_kl_validation(self):
  b,Q,H,W=preference_problem();g=GuardedEngine(b,Q,H,kappa=.05)
  self.assertEqual(g.run(lambda e,t:1,3)['labels'],3)
  with self.assertRaises(ValueError):GuardedEngine(b,Q,H,kappa=.001)
 def test_null_never_accepts_in_fixed_regression_paths(self):
  b,Q,H,W=preference_problem()
  for seed in range(8):
   rng=np.random.default_rng(seed);g=GuardedEngine(b,Q,H,prune=True)
   v=g.run(lambda e,t:int(rng.integers(2)),200)
   self.assertFalse(v['certified'])
 def test_drift_invalidates_stale_certificates(self):
  b,Q,H,W=preference_problem();g=GuardedEngine(b,Q,H,rho=.001)
  g.run(lambda e,t:1,20)
  before=g.result()['simple_regret_certificate'];g.advance(10000)
  self.assertTrue(np.all(g.lo==-1));self.assertTrue(np.all(g.hi==1))
  self.assertGreaterEqual(g.result()['simple_regret_certificate'],before)
  self.assertIsNone(g.result()['rate_bound'])
  with self.assertRaises(ValueError):g.observe(0,1)
 def test_all_zero_influence(self):
  b=np.ones(3)/3;g=GuardedEngine(b,[b],np.eye(3),prune=True)
  v=g.run(lambda e,t:self.fail('oracle should not be queried'),100)
  self.assertEqual(v['labels'],0);self.assertFalse(v['certified'])
 def test_truth_covered_regret_certificate(self):
  b,Q,H,W=preference_problem()
  edges=[(i,j) for i in range(5) for j in range(i+1,5)]
  rng=np.random.default_rng(45);g=GuardedEngine(b,Q,H)
  v=g.run(lambda e,t:int(rng.binomial(1,(1+W[edges[e]])/2)),600)
  gains=np.array([min((q-b)@W@H.T) for q in Q])
  self.assertLessEqual(max(gains)-gains[v['policy_index']],v['simple_regret_certificate'])
if __name__=='__main__':unittest.main()

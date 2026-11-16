import unittest
import numpy as np
from exact_budget_gate import Gate
from guarded_engine import GuardedEngine
from rational_check import solve

class CoreGateTests(unittest.TestCase):
 def test_embedding_and_kl(self):
  b=np.ones(3)/3;Q=np.array([b+.12*np.array([-1,1,0]),b+.12*np.array([-1,0,1])]);H=np.array([[1.,0.,0.]])
  self.assertTrue(np.all(np.sum(Q*np.log(Q/b),axis=1)<.05))
  rng=np.random.default_rng(17)
  for _ in range(20):
   p=rng.random(3);W=np.zeros((3,3))
   for e,(i,j) in enumerate(((0,1),(0,2),(1,2))):W[i,j]=2*p[e]-1;W[j,i]=-W[i,j]
   np.testing.assert_allclose((Q-b)@W@H.T,.12*(1-2*p[:2,None]),rtol=0,atol=1e-15)
 def test_rule_matches_frozen_engine(self):
  gate=Gate();b=np.ones(3)/3;Q=np.array([b+.12*np.array([-1,1,0]),b+.12*np.array([-1,0,1])]);H=np.array([[1.,0.,0.]])
  rng=np.random.default_rng(55)
  for _ in range(200):
   n0=int(rng.integers(1,12));n1=int(rng.integers(1,12));s0=int(rng.integers(n0+1));s1=int(rng.integers(n1+1))
   st=(n0,s0,n1,s1)
   if gate.terminal(st):continue
   g=GuardedEngine(b,Q,H,prune=True,incremental=False)
   g.n[:2]=[n0,n1];g.wins[:2]=[s0,s1];g.labels=n0+n1;g.rr=(g.labels-2)//20
   for e,(n,s) in enumerate(((n0,s0),(n1,s1))):g.lo[e],g.hi[e]=gate.intervals[n,s]
   g.bounds()
   self.assertEqual(g.next_edge(),gate.heuristic(st,'joint_mix'))
 def test_forward_backward_and_rational(self):
  r=solve();g=Gate()
  for m in ('greedy','joint_mix','bellman'):
   self.assertAlmostEqual(g.distribution(m)['success_probability'],r[m]['decimal'],places=13)
  self.assertTrue(r['strict_bellman_advantage']);self.assertTrue(r['strict_greedy_advantage'])
 def test_fixed_null_error_and_mass(self):
  g=Gate()
  for m in ('greedy','joint_mix','bellman'):
   v=g.distribution(m,(.5,.5))
   self.assertAlmostEqual(v['terminal_mass'],1,places=12)
   self.assertLessEqual(v['false_certificate_probability'],.05)
if __name__=='__main__':unittest.main()

import unittest,random
from margin_horizon import plan,certificate,allocations
class TestPlanner(unittest.TestCase):
 def test_exact(self):
  rng=random.Random(72)
  for _ in range(100):
   E=rng.randint(2,4); B=rng.randint(0,6)
   scores=[[rng.uniform(-.2,.2) for _ in range(3)] for q in range(2)]
   weights=[[[rng.random() for e in range(E)] for h in range(3)] for q in range(2)]
   n=[rng.randint(1,8) for e in range(E)]; c=[rng.random() for e in range(E)]
   v,x=plan(scores,weights,n,c,B)
   oracle=max(certificate(scores,weights,n,z,c) for z in allocations(E,B))
   self.assertAlmostEqual(v,oracle)
   self.assertEqual(sum(x),B)
 def test_policy_switch(self):
  scores=[[.2],[.15]];weights=[[[2.,0.]],[[0.,.1]]]
  v,x=plan(scores,weights,[1,1],[1,1],3)
  self.assertAlmostEqual(v,certificate(scores,weights,[1,1],x,[1,1]))
 def test_bad(self):
  with self.assertRaises(ValueError):plan([[-1]],[[[1]]],[0],[1],2)
if __name__=='__main__':unittest.main()

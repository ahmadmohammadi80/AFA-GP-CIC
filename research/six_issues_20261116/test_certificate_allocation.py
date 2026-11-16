import unittest,random
from certificate_allocation import choose_minimax_edge,exhaustive_oracle,radius
class TestMACWA(unittest.TestCase):
 def test_exhaustive_oracle(self):
  rng=random.Random(20261116)
  for _ in range(300):
   E=rng.randint(2,12);R=rng.randint(1,8)
   A=[[rng.uniform(-1,1) for _ in range(E)] for _ in range(R)]
   n=[rng.randint(1,1000) for _ in range(E)]
   active=rng.sample(range(R),rng.randint(1,R))
   e,v,cs=choose_minimax_edge(A,n,active=active)
   f,fs=exhaustive_oracle(A,n,active=active)
   self.assertEqual(e,f)
   for x,y in zip(cs,fs):self.assertAlmostEqual(x,y,places=12)
 def test_monotonicity(self):
  A=[[.1,.8,.2],[.9,.2,.1]];n=[50,40,100]
  _,after,_=choose_minimax_edge(A,n)
  before=max(sum(abs(a)*radius(k,.05,3) for a,k in zip(row,n)) for row in A)
  self.assertLessEqual(after,before+1e-12)
 def test_invalid(self):
  with self.assertRaises(ValueError):choose_minimax_edge([[1,2]],[1])
  with self.assertRaises(ValueError):choose_minimax_edge([[1,2]],[0,2])
if __name__=="__main__":unittest.main()

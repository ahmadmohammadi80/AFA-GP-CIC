import unittest,itertools
import numpy as np
from certificate_budget_oracle import allocate,integer_allocation,optimal_width
class TestBudget(unittest.TestCase):
 def test_closed_form(self):
  for seed in range(30):
   r=np.random.default_rng(seed);w=r.uniform(.01,2,size=5);B=20
   n,v=allocate(w,B)
   self.assertAlmostEqual(n.sum(),B)
   self.assertAlmostEqual(v,optimal_width(w,B),places=11)
   self.assertLessEqual(v,np.sum(w/np.sqrt(B/len(w)))+1e-9)
 def test_integer_exact_bruteforce(self):
  for w in ([2.,1.,.5],[0.,1.,3.],[1.,1.,1.]):
   n,v=integer_allocation(w,9)
   active=[i for i,x in enumerate(w) if x>0]
   vals=[]
   for choices in itertools.product(range(1,10),repeat=len(active)):
    if sum(choices)==9:vals.append(sum(w[e]/np.sqrt(c) for e,c in zip(active,choices)))
   self.assertAlmostEqual(v,min(vals))
   self.assertEqual(n.sum(),9)
 def test_zero_and_invalid(self):
  self.assertEqual(allocate([0.,0.],10)[1],0.)
  with self.assertRaises(ValueError):integer_allocation([1.,2.],1)
if __name__=="__main__":unittest.main()

"""Sharp finite-horizon coupling bound for fixed-parameter adaptive Bernoulli queries.
For ||p-q||_infinity<=r and <=B queries, |P_p(C)-P_q(C)|<=1-(1-r)**B.
This is a standard coupling refinement, NOT a global novelty claim.
"""
import unittest
from itertools import product
from math import isclose

def coupling_penalty(budget, radius):
    if not isinstance(budget,int) or budget<0 or not 0<=radius<=1:
        raise ValueError('invalid budget/radius')
    return -__import__('math').expm1(budget*__import__('math').log1p(-radius)) if radius<1 else (float(budget>0))

def certified_lower(grid_minimum,budget,radius):
    if not 0<=grid_minimum<=1: raise ValueError('invalid grid probability')
    return max(0.,grid_minimum-coupling_penalty(budget,radius))

class TestSharpCoupling(unittest.TestCase):
    def test_tight_any_success(self):
        for B in range(1,9):
            for r in (.01,.1,.25,.5,.9):
                exact=1-(1-r)**B
                self.assertAlmostEqual(coupling_penalty(B,r),exact,places=12)
    def test_stronger_than_linear(self):
        for B,r in product(range(0,12),(.0,.01,.2,.5,1.)):
            d=coupling_penalty(B,r)
            self.assertLessEqual(d,B*r+1e-12)
            self.assertGreaterEqual(d,0)
    def test_invalid(self):
        for B,r in ((-1,.2),(1,-.1),(2,1.1),(1,float('nan'))):
            with self.assertRaises(ValueError):coupling_penalty(B,r)
    def test_lower(self):
        self.assertAlmostEqual(certified_lower(.7,2,.1),.51)
        self.assertEqual(certified_lower(.1,3,.5),0.)
if __name__=='__main__':unittest.main()

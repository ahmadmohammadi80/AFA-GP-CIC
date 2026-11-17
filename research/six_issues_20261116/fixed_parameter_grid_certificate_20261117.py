"""Fixed-parameter grid certificate for a finite family of adaptive query policies.
For each fixed policy pi, the probability of reaching certificate C by B queries
is B-Lipschitz in ||p-q||_infinity via sequential Bernoulli coupling.
Consequently min_grid P_pi - B*r <= inf_box P_pi <= min_grid P_pi,
where r is the L-infinity grid covering radius. Selecting the largest grid
minimum among candidate policies yields true worst-case success at least
OPT_candidates - B*r. This is a generic discretization result, NOT a claim
of global scientific novelty. C must be an externally validated certificate.
"""
from itertools import product
from functools import lru_cache
import unittest

def evaluate(policy, p, budget, certified):
    k=len(p)
    @lru_cache(None)
    def rec(n,s,t):
        if certified(n,s): return 1.
        if t==0: return 0.
        e=policy(n,s,t)
        if not isinstance(e,int) or not 0<=e<k: raise ValueError("invalid edge")
        nn=list(n); nn[e]+=1
        ss=list(s); ss[e]+=1
        return p[e]*rec(tuple(nn),tuple(ss),t-1)+(1-p[e])*rec(tuple(nn),s,t-1)
    return rec((0,)*k,(0,)*k,budget)

def select(policies, intervals, budget, certified, steps=10):
    if not policies or budget<0 or steps<1: raise ValueError("invalid setup")
    if not intervals or any(not 0<=a<=b<=1 for a,b in intervals):
        raise ValueError("invalid intervals")
    axes=[[a+(b-a)*i/steps for i in range(steps+1)] for a,b in intervals]
    radius=max((b-a)/(2*steps) for a,b in intervals)
    grid=list(product(*axes))
    scores=[min(evaluate(policy,p,budget,certified) for p in grid) for policy in policies]
    idx=max(range(len(scores)),key=lambda i:(scores[i],-i))
    return idx,max(0.,scores[idx]-budget*radius),scores[idx],radius

class TestFixedGrid(unittest.TestCase):
    def test_fixed_nonrectangular_example(self):
        policy=lambda n,s,t:0
        cert=lambda n,s:n[0]==2 and s[0]==1
        idx,lower,score,r=select([policy],[(.2,.8)],2,cert,6)
        self.assertEqual(idx,0)
        self.assertAlmostEqual(score,.32)
        self.assertLessEqual(lower,.32)
        self.assertAlmostEqual(evaluate(policy,(.5,),2,cert),.5)
    def test_covering_grid(self):
        policies=[lambda n,s,t:0,lambda n,s,t:1,
                  lambda n,s,t:(n[0]+n[1])%2]
        cert=lambda n,s:sum(s)>=2
        idx,lower,score,r=select(policies,[(.2,.7),(.1,.9)],3,cert,4)
        self.assertTrue(0<=lower<=score<=1)
        for i in range(21):
            for j in range(21):
                p=(.2+.5*i/20,.1+.8*j/20)
                self.assertGreaterEqual(evaluate(policies[idx],p,3,cert)+1e-12,lower)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            select([],[(0,1)],2,lambda n,s:False)
        with self.assertRaises(ValueError):
            select([lambda n,s,t:0],[(1,0)],2,lambda n,s:False)

if __name__=="__main__": unittest.main()
